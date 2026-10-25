"""KION winner features plus rotation-invariant hybrid-factor contributions."""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from .v2_features import TemporalFeatureStore, feature_request, select_columns


class WinnerFeatureStore(TemporalFeatureStore):
    def __init__(self,events,items,users,index,windows=None,representation_model=None,required_columns=None):
        super().__init__(events,items,users,index,windows,required_columns=required_columns)
        self.representation_model=representation_model
        grouped=pd.DataFrame({'item_id':self.history.item_id.to_numpy(),'age':self.age}).groupby('item_id').age
        self.item_median=grouped.median().reindex(index.items).to_numpy(np.float32)
        self.item_std=grouped.std().reindex(index.items).fillna(0).to_numpy(np.float32)
        day=np.ceil(self.age).astype(int)
        recent=(day>=1)&(day<=7)
        daily=np.bincount(self.hi[recent]*7+(7-day[recent]),minlength=len(index.items)*7).reshape(-1,7)
        extended=np.c_[2*daily[:,0:1]-daily[:,6::-1],daily,2*daily[:,-1:]-daily[:,-1:-7:-1]]
        weights=np.hamming(7);weights/=weights.sum()
        smoothed=convolve1d(extended.astype(float),weights,axis=1,mode='constant')[:,7:-6]
        self.hamming_trend=(smoothed[:,-1]-smoothed[:,-2]).astype(np.float32)
        self.recent_fraction={}
        for fraction in [.01,.05,.1,.2,.4,.8]:
            threshold=np.quantile(self.age,fraction)
            # Complete timestamp ties: no arbitrary ordering within a calendar day.
            mask=self.age<=threshold
            self.recent_fraction[int(fraction*100)]=np.bincount(self.hi[mask],minlength=len(index.items)).astype(np.float32)
        self.catalog_features={}
        for relation in ['genre','country']:
            matrix=self.token_matrices.get(relation)
            if matrix is None:continue
            counts=np.asarray(matrix.sum(axis=0)).ravel()
            minimum=[];maximum=[];median=[]
            for row in range(len(index.items)):
                token=matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
                values=counts[token]
                minimum.append(values.min() if len(values) else 0)
                maximum.append(values.max() if len(values) else 0)
                median.append(np.median(values) if len(values) else 0)
            for op,values in [('min',minimum),('max',maximum),('median',median)]:
                self.catalog_features[f'catalog_{relation}_{op}']=np.asarray(values,np.float32)
        if 'studios' in items:
            tokens=[sorted(set(t.strip() for t in value.split(',') if t.strip())) or ['unknown']
                    for value in items.studios.reindex(index.items).fillna('unknown').astype(str)]
            count=Counter(t for row in tokens for t in row)
            self.catalog_features['catalog_studios_max']=np.asarray([max(count[t] for t in row) for row in tokens],np.float32)

    def build(self,candidates,with_auto=True,with_history=True,with_embeddings=True,required_columns=None):
        required_columns, with_embeddings = feature_request(self, required_columns, with_embeddings)
        # Individual latent coordinates are rotation-dependent across prefix fits.
        # This variant exposes stable scalar blocks instead of arbitrary axes.
        frame,cats,groups=super().build(candidates,with_auto,with_history,False,required_columns=required_columns)
        iid=self.item_pos.get_indexer(candidates.item_id)
        extras={
            'item_median_age':self.item_median[iid],
            'item_date_std':self.item_std[iid],
            'item_hamming_trend7':self.hamming_trend[iid],
            **{name:values[iid] for name,values in self.catalog_features.items()},
            **{f'item_recent_fraction_{fraction}':values[iid] for fraction,values in self.recent_fraction.items()},
        }
        groups['manual'].extend(extras)
        if with_embeddings and self.representation_model is not None:
            model=self.representation_model
            uid=self.user_pos.get_indexer(candidates.user_id)
            du=len(model.user_feature_names);di=len(model.item_feature_names)
            together=model.config.get('fit_features_together',False)
            latent=int(model.config.get('factors',32))
            numeric=defaultdict(lambda:np.zeros(len(candidates),np.float32))
            if together:
                # Named, fixed demographic blocks are invariant to latent rotations.
                for j,name in enumerate(model.user_feature_names):
                    field=name.split('=',1)[0]
                    numeric['learned_demographic_'+field] += model.user_factors[uid,j]*model.item_factors[iid,j]
                offset=du+latent
                for j,name in enumerate(model.item_feature_names):
                    field=name.split('=',1)[0]
                    numeric['learned_content_'+field] += model.user_factors[uid,offset+j]*model.item_factors[iid,offset+j]
                begin,end=du,du+latent
            else:
                begin,end=0,latent
            dot=np.zeros(len(candidates),np.float32)
            unorm=np.zeros(len(candidates),np.float32)
            inorm=np.zeros(len(candidates),np.float32)
            for dim in range(begin,end):
                uf=model.user_factors[uid,dim];it=model.item_factors[iid,dim]
                dot +=uf*it;unorm+=uf*uf;inorm+=it*it
            numeric['learned_latent_dot']=dot
            numeric['learned_latent_cosine']=dot/np.maximum(np.sqrt(unorm*inorm),1e-12)
            numeric['learned_user_norm']=np.sqrt(unorm)
            numeric['learned_item_norm']=np.sqrt(inorm)
            extras.update(numeric)
            groups['latent']=list(numeric)
        result = pd.concat([frame,pd.DataFrame(extras,index=frame.index)],axis=1)
        return select_columns(result,cats,groups,required_columns)
