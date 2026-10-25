"""Past-only attributes of completed, last-dated KION aggregates.

For a retained row, last_watch_dt < cutoff implies every contributing watch is
in the past. Durations here are lifetime totals of those completed pairs, NOT
watch time consumed solely within an aggregation window. No future row can be
partially recovered. This relies on benchmark event-date availability, not an
undocumented production ingestion SLA.
"""
import numpy as np
import pandas as pd
from .v3_features import WinnerFeatureStore
from .v2_features import MANUAL_WINDOWS, required_windows, feature_request, select_columns


class QualityWinnerFeatureStore(WinnerFeatureStore):
    def __init__(self,events,items,users,index,windows=None,representation_model=None,required_columns=None):
        if not {'total_dur','watched_pct'}.issubset(events):
            raise ValueError('Completed-pair quality features require original aggregate attributes')
        if representation_model is not None:
            if not np.array_equal(representation_model.users,index.users) or not np.array_equal(representation_model.items,index.items):
                raise ValueError('Representation IDs must align with the raw prefix index')
            if hasattr(representation_model,'cutoff') and pd.Timestamp(representation_model.cutoff)!=pd.Timestamp(index.cutoff):
                raise ValueError('Representation cutoff must match the feature cutoff')
        super().__init__(events,items,users,index,windows,representation_model,required_columns=required_columns)
        past=events.loc[events.timestamp<self.cutoff]
        duration=pd.to_numeric(past.total_dur,errors='coerce').fillna(0).clip(lower=0).to_numpy()
        percentage=pd.to_numeric(past.watched_pct,errors='coerce').fillna(0).clip(0,1000).to_numpy()
        # Match the original RecTools threshold boundary; percentage>10 is used
        # for confidence weighting in the model, not for changing our labels.
        engaged=duration>=300
        self.quality_counts={}
        for window in self.windows:
            mask=engaged & (np.ones(len(self.age),bool) if not window else self.age<=window)
            self.quality_counts[window]={
                'user':np.bincount(self.hu[mask],minlength=len(index.users)).astype(np.float32),
                'item':np.bincount(self.hi[mask],minlength=len(index.items)).astype(np.float32)}
        self.quality_means={}
        for entity,pos,size,denominator in [
            ('user',self.hu,len(index.users),self.user_stats[0]),
            ('item',self.hi,len(index.items),self.item_stats[0])]:
            for name,values in [('log_pair_duration',np.log1p(duration)),('pair_watched_pct',percentage),
                                ('strong_pair',engaged & (percentage>10))]:
                self.quality_means[entity+'_'+name]=(
                    np.bincount(pos,weights=values,minlength=size)/np.maximum(denominator,1)).astype(np.float32)

    def build(self,candidates,with_auto=True,with_history=True,with_embeddings=True,required_columns=None):
        required_columns, with_embeddings = feature_request(self, required_columns, with_embeddings)
        frame,cats,groups=super().build(candidates,with_auto,with_history,with_embeddings,required_columns=required_columns)
        uid=self.user_pos.get_indexer(candidates.user_id)
        iid=self.item_pos.get_indexer(candidates.item_id)
        extra={}
        if with_embeddings and self.representation_model is not None:
            model=self.representation_model
            for entity,pos,size in [('user',uid,len(self.index.users)),('item',iid,len(self.index.items))]:
                active=getattr(model,'active_'+entity+'s',np.ones(size,bool))
                name='learned_'+entity+'_active'
                extra[name]=np.asarray(active,dtype=np.float32)[pos]
                groups.setdefault('latent',[]).append(name)
        for window in self.windows:
            if required_columns is not None and window not in MANUAL_WINDOWS + required_windows(required_columns):continue
            if not with_auto and window not in MANUAL_WINDOWS:continue
            group='manual' if window in MANUAL_WINDOWS else f'window_{window}'
            tag=str(window or 'all')
            for entity,pos,counts in [('user',uid,self.user_stats),('item',iid,self.item_stats)]:
                values=self.quality_counts[window][entity][pos]
                name=f'{entity}_completed_engaged_count_{tag}'
                extra[name]=values
                extra[name+'_share']=values/np.maximum(counts[window][pos],1)
                groups.setdefault(group,[]).extend([name,name+'_share'])
        for name,values in self.quality_means.items():
            pos=uid if name.startswith('user_') else iid
            extra[name]=values[pos]
            groups['manual'].append(name)
        result = pd.concat([frame,pd.DataFrame(extra,index=frame.index)],axis=1)
        return select_columns(result,cats,groups,required_columns)
