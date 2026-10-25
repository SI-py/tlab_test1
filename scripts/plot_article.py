"""Static publication figures from recorded experiment aggregates only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/tlab1-article-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

NAMES={'manual':'Ручная база','auto_windows':'Все окна','auto_selected':'Выбранные окна',
       'representations':'Представления','auto_full':'Окна + представления',
       'anchor_representations':'Представления + якоря','anchor_auto_full':'Окна + якоря',
       'primary':'Исходная смесь KION','popularity':'Популярность','rank_fusion':'RRF генераторов'}
ORDER=list(NAMES)

def metrics(data,key):
    value=data.get(key,{})
    return value.get('metrics',value)

def plots(data,output,random=None,serving=None,yambda=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    font=Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf')
    if font.exists():
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman' if font.exists() else 'DejaVu Serif',
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250,
        'svg.hashsalt':'tlab1-article','axes.axisbelow':True})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=250,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
        plt.close(fig)
    val=metrics(data,'validation');test=metrics(data,'test')
    names=[name for name in ORDER if name in test and name in val]
    fig,axs=plt.subplots(1,2,figsize=(6.6,3.65),layout='constrained')
    for ax,key,title in [(axs[0],'ndcg@10','NDCG@10'),(axs[1],'recall@10','Recall@10')]:
        y=np.arange(len(names))
        ax.plot([val[name][key] for name in names],y-.10,'o',color='#8c8c8c',ms=4,label='Validation, seed 17')
        ax.plot([test[name][key] for name in names],y+.10,'s',color='#285a7b',ms=4,label='Confirmation')
        ax.set(yticks=y,yticklabels=[NAMES[name] for name in names] if ax is axs[0] else [],title=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.22)
        ax.tick_params(axis='y',length=0)
    axs[1].legend(loc='lower right',fontsize=7,frameon=False)
    save(fig,'quality')
    comparisons=[]
    for name,by in data.get('paired_bootstrap',{}).items():
        if name=='manual_vs_popularity':continue
        comparisons.append((NAMES.get(name,name)+' - ручная база',by))
    for record in data.get('diagnostic_paired_bootstrap',{}).values():
        comparisons.append((NAMES.get(record['challenger'],record['challenger'])+' - '+
                            NAMES.get(record['reference'],record['reference']),record['bootstrap']))
    fig,axs=plt.subplots(1,2,figsize=(6.6,max(2.2,.37*len(comparisons)+.7)),layout='constrained')
    for ax,key,title in [(axs[0],'ndcg','ΔNDCG@10, п.п.'),(axs[1],'recall','ΔRecall@10, п.п.')]:
        for row,(label,by) in enumerate(comparisons):
            lo,hi=np.array(by[key]['ci95'])*100
            ax.hlines(row,lo,hi,color='#285a7b',lw=1.4)
            ax.plot(by[key]['delta']*100,row,'o',color='#285a7b',ms=4)
        ax.axvline(0,color='.5',ls='--',lw=.8)
        ax.set(yticks=range(len(comparisons)),yticklabels=[name for name,_ in comparisons] if ax is axs[0] else [],xlabel=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
    save(fig,'paired_deltas')
    if random and random.get('status')=='complete':
        fig,ax=plt.subplots(figsize=(6.6,2.25),layout='constrained')
        for method,label,color,marker in [('importance','Importance','#285a7b','o'),
                                          ('random','Random, seed 20261025','#a87545','s')]:
            trials=random[method]['trials']
            ax.plot([t['cardinality'] for t in trials],[t['metrics']['ndcg@10'] for t in trials],
                    marker=marker,color=color,lw=1.3,ms=4,label=label)
        ax.set(xticks=[1,2,3,5],xlabel='Число дополнительных окон',ylabel='Development NDCG@10')
        ax.grid(alpha=.22);ax.legend(frameon=False,fontsize=8)
        save(fig,'random_control')
    if serving:
        keys=['manual','auto_selected','primary'];cases=serving['cases']
        fig,axs=plt.subplots(1,2,figsize=(6.6,2.7),layout='constrained')
        for ax,key,title in [(axs[0],'feature_only','Только selective build'),
                             (axs[1],'full_warm_request','Полный warm request')]:
            means=np.array([cases[k][key]['mean_seconds'] for k in keys]);y=np.arange(3)
            low=means-np.array([cases[k][key]['min_seconds'] for k in keys])
            high=np.array([cases[k][key]['max_seconds'] for k in keys])-means
            ax.barh(y,means,color=['#737373','#285a7b','#a87545'],height=.55)
            ax.errorbar(means,y,xerr=np.array([low,high]),fmt='none',ecolor='black',capsize=3,lw=.8)
            ax.set(yticks=y,yticklabels=[NAMES[k] for k in keys] if ax is axs[0] else [],
                   xlabel='Секунды на batch 128 пользователей',title=title,xlim=(0,max(means)*1.22))
            for pos,mean in enumerate(means):ax.text(mean+max(means)*.025,pos,f'{mean:.3f}',va='center',fontsize=8)
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
        save(fig,'serving_cost')
    if yambda and yambda.get('status')=='complete':
        keys=['manual','importance','random'];labels=['Manual','Importance','Random']
        weeks=yambda['test_weeks'];days=[w['day'] for w in weeks]
        fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),layout='constrained')
        for ax,key,title in [(axs[0],'ndcg','NDCG@10'),(axs[1],'recall','Recall@10')]:
            for arm,label,color,marker in zip(keys,labels,['#737373','#285a7b','#a87545'],['o','s','^']):
                ax.plot(days,[w['systems'][arm]['all_active'][key] for w in weeks],
                        marker=marker,color=color,label=label,lw=1.3,ms=4)
            ax.set(xticks=days,xlabel='Относительный день начала недели',title=title)
            ax.grid(alpha=.22)
        axs[1].legend(frameon=False,fontsize=8)
        save(fig,'yambda_weekly')
        contrasts=[('importance_minus_manual','Importance - manual'),
                   ('random_minus_manual','Random - manual'),('importance_minus_random','Importance - random')]
        paired=yambda['paired_user_bootstrap']['all_active']
        fig,axs=plt.subplots(1,2,figsize=(6.6,2.4),layout='constrained')
        for ax,key,title in [(axs[0],'ndcg','ΔNDCG@10, п.п.'),(axs[1],'recall','ΔRecall@10, п.п.')]:
            for row,(name,label) in enumerate(contrasts):
                record=paired[name][key];lo,hi=np.asarray(record['ci95'])*100
                ax.hlines(row,lo,hi,color='#285a7b',lw=1.3)
                ax.plot(record['delta']*100,row,'o',color='#285a7b',ms=4)
            ax.axvline(0,color='.5',ls='--',lw=.8)
            ax.set(yticks=range(3),yticklabels=[label for _,label in contrasts] if ax is axs[0] else [],xlabel=title)
            ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
        save(fig,'yambda_deltas')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/experiment.json'))
    parser.add_argument('--output',type=Path,default=Path('results/figures'))
    parser.add_argument('--random',type=Path)
    parser.add_argument('--serving',type=Path)
    parser.add_argument('--yambda',type=Path)
    args=parser.parse_args()
    def optional(path,name):
        path=path or args.input.parent/name
        return json.loads(path.read_text()) if path.exists() else None
    plots(json.loads(args.input.read_text()),args.output,optional(args.random,'random_search.json'),
          optional(args.serving,'serving_benchmark.json'),optional(args.yambda,'yambda.json'))

if __name__=='__main__':main()
