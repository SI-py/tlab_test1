"""Build the Russian research article from completed aggregate results only.

Run from repository root: python scripts/article.py --input results/experiment.json
Requires reportlab and the publication figures produced by plot_article.py.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image,PageBreak,Paragraph,SimpleDocTemplate,Spacer,Table,TableStyle,KeepTogether

TITLE='Автоматический выбор временных признаков и обучаемые представления в рекомендательных системах'
NAMES={'manual':'Ручная база','auto_windows':'Все окна','auto_selected':'Выбранные окна',
 'representations':'Представления','auto_full':'Окна + представления',
 'anchor_representations':'Представления + якоря','anchor_auto_full':'Окна + якоря',
 'primary':'Исходная смесь KION','popularity':'Популярность','rank_fusion':'RRF генераторов'}
ORDER=['manual','auto_windows','auto_selected','representations','auto_full',
       'anchor_representations','anchor_auto_full','rank_fusion','primary','popularity']

def num(value,digits=4):return 'не измерено' if value is None else f'{value:.{digits}f}'
def signed(value,digits=4):return f'{value:+.{digits}f}'
def ci(item):return f"{signed(item['delta'],6)} [{signed(item['ci95'][0],6)}; {signed(item['ci95'][1],6)}]"
def metric_section(data,name):
    section=data.get(name,{})
    return section.get('metrics',section)
def decision_metrics(entry):return entry.get('metrics',entry)
def change_text(item):
    low,high=item['ci95']
    if low>0:return 'интервал полностью выше нуля'
    if high<0:return 'интервал полностью ниже нуля'
    return 'интервал включает ноль'

def fonts():
    directory=Path('/System/Library/Fonts/Supplemental')
    files=['Times New Roman.ttf','Times New Roman Bold.ttf','Times New Roman Italic.ttf','Times New Roman Bold Italic.ttf']
    if not (directory/files[0]).exists():
        directory=Path('/usr/share/fonts/truetype/dejavu')
        files=['DejaVuSerif.ttf','DejaVuSerif-Bold.ttf','DejaVuSerif-Italic.ttf','DejaVuSerif-BoldItalic.ttf']
    for name,file in zip(['Article','ArticleB','ArticleI','ArticleBI'],files):pdfmetrics.registerFont(TTFont(name,str(directory/file)))
    pdfmetrics.registerFontFamily('Article',normal='Article',bold='ArticleB',italic='ArticleI',boldItalic='ArticleBI')

class Article:
    def __init__(self,input,output,random=None,serving=None,yambda=None,figures=None):
        fonts();self.input=Path(input);self.output=Path(output);self.data=json.loads(self.input.read_text())
        def recorded(path,name):
            path=Path(path) if path else self.input.parent/name
            return json.loads(path.read_text()) if path.exists() else None
        self.random=recorded(random,'random_search.json')
        self.serving=recorded(serving,'serving_benchmark.json')
        self.yambda=recorded(yambda,'yambda.json')
        if not self.random or self.random.get('status')!='complete':raise ValueError('Completed random control required')
        if not self.serving or set(self.serving.get('cases',{}))!={'manual','auto_selected','primary'}:
            raise ValueError('Completed three-system serving benchmark required')
        if not all(c.get('feature_parity_exact') for c in self.serving['cases'].values()):
            raise ValueError('Serving feature parity must be established')
        if not self.yambda or self.yambda.get('status')!='complete':raise ValueError('Completed external Yambda results required')
        self.val={name:decision_metrics(value) for name,value in metric_section(self.data,'validation').items()}
        self.test=metric_section(self.data,'test');self.choices=self.data['model_choices']
        self.paired=self.data.get('paired_bootstrap',{});self.diag=self.data.get('diagnostic_paired_bootstrap',{})
        mix=self.data['primary_mixture'];self.mix=mix.get('weights',mix)
        self.figures=Path(figures) if figures else self.input.parent/'figures'
        self.width=A4[0]-44*mm;self.story=[]
        self.section_number=self.table_number=self.figure_number=0
        self.styles={
          'title':ParagraphStyle('title',fontName='ArticleB',fontSize=20,leading=24,alignment=TA_CENTER,spaceAfter=15),
          'subtitle':ParagraphStyle('subtitle',fontName='Article',fontSize=10,leading=13,alignment=TA_CENTER,spaceAfter=14),
          'body':ParagraphStyle('body',fontName='Article',fontSize=10.6,leading=13.8,alignment=TA_JUSTIFY,spaceAfter=8),
          'abstract':ParagraphStyle('abstract',fontName='Article',fontSize=10.1,leading=13.2,alignment=TA_JUSTIFY,leftIndent=10,rightIndent=10,spaceAfter=12),
          'heading':ParagraphStyle('heading',fontName='ArticleB',fontSize=13.5,leading=17,spaceBefore=5,spaceAfter=9,keepWithNext=True),
          'sub':ParagraphStyle('sub',fontName='ArticleB',fontSize=11.2,leading=14,spaceBefore=5,spaceAfter=7,keepWithNext=True),
          'small':ParagraphStyle('small',fontName='Article',fontSize=9.1,leading=11.6,spaceAfter=8),
          'cell':ParagraphStyle('cell',fontName='Article',fontSize=8.3,leading=10.5),
          'head':ParagraphStyle('head',fontName='ArticleB',fontSize=8.3,leading=10.5),
          'equation':ParagraphStyle('equation',fontName='ArticleI',fontSize=10.4,leading=17,alignment=TA_CENTER,spaceBefore=3,spaceAfter=9),
        }
    def p(self,text,style='body'):
        for old in ['\u2011','\u2013','\u2014']:text=text.replace(old,'-')
        self.story.append(Paragraph(text,self.styles[style]))
    def h(self,text):
        self.section_number+=1
        self.p(str(self.section_number)+'. '+re.sub(r'^\d+\.\s*','',text),'heading')
    def sub(self,text):self.p(text,'sub')
    def eq(self,text):self.p(text,'equation')
    def page(self):self.story.append(PageBreak())
    def pair(self,reference,challenger):
        if reference=='manual' and challenger in self.paired:return self.paired[challenger]
        for record in self.diag.values():
            if record['reference']==reference and record['challenger']==challenger:return record['bootstrap']
        return None
    def finding(self,reference,challenger):
        delta=self.test[challenger]['ndcg@10']-self.test[reference]['ndcg@10']
        pair=self.pair(reference,challenger)
        case={'manual':'ручной базы','representations':'представлений','auto_full':'окон с представлениями'}
        text=f"{NAMES[challenger]} относительно {case.get(reference,NAMES[reference].lower())}: ΔNDCG@10 {signed(delta,6)}"
        if pair:
            d=pair['ndcg'];text+=f", 95% интервал [{signed(d['ci95'][0],6)}; {signed(d['ci95'][1],6)}]"
            text+='; '+change_text(d)
        return text+'.'
    def table(self,headers,rows,widths,caption):
        self.table_number+=1
        self.p('Таблица '+str(self.table_number)+'. '+re.sub(r'^Таблица\s+\d+\.\s*','',caption),'small')
        content=[[Paragraph(escape(str(x)),self.styles['head']) for x in headers]]
        content.extend([[Paragraph(escape(str(x)),self.styles['cell']) for x in row] for row in rows])
        t=Table(content,colWidths=[self.width*w for w in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),
          ('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
          ('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
          ('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.Color(.968,.968,.968)])]))
        self.story.extend([t,Spacer(1,7)])
    def fig(self,name,height,caption):
        path=self.figures/name
        if not path.exists():raise FileNotFoundError('Build publication figures first: '+str(path))
        self.story.append(Image(str(path),width=self.width,height=height,kind='proportional'))
        self.figure_number+=1
        self.p('Рисунок '+str(self.figure_number)+'. '+re.sub(r'^Рисунок\s+\d+\.\s*','',caption),'small')
    def title(self):
        self.p(TITLE,'title')
        self.p('KION и Yambda: сильная ручная база, равный бюджет поиска и стоимость запроса','subtitle')
        primary=self.test['primary'];manual=self.test['manual'];delta=primary['ndcg@10']-manual['ndcg@10']
        inference=self.paired.get('primary',{}).get('ndcg')
        interval=('; 95% пользовательский bootstrap: ['+signed(inference['ci95'][0],6)+'; '+signed(inference['ci95'][1],6)+']') if inference else ''
        abstract=('<b>Аннотация.</b> Проверяется ограниченная автоматизация временных признаков поверх сильного '
          'двухуровневого рекомендателя. На KION все ранжировщики получают общий union кандидатов; '
          'дополнительные окна дополняют ручную базу, а компактные ALS-признаки и якоря проверяют другую '
          'гипотезу - частичную замену агрегатов. У выбранных окон итоговый NDCG@10 равен '+
          num(self.test['auto_selected']['ndcg@10'],6)+' против '+num(manual['ndcg@10'],6)+
          ' у manual; календарная неделя ранее изучалась на других пользователях. '
          'Позднее добавлен контроль четырёх случайных наборов окон при том же условном бюджете четырёх обучений: '
          'его лучший development NDCG ниже importance-процедуры на '+
          num(-self.random['comparison']['random_minus_importance_ndcg@10'],6)+'. ')
        result=self.yambda['primary_result'];b=result['bootstrap']
        abstract+=('Внешняя проверка на Yambda использует новые когорты и три заранее заданные недели: '
          'importance-отбор относительно manual даёт ΔNDCG '+signed(b['delta'],6)+
          ', пользовательский 95% интервал ['+signed(b['ci95'][0],6)+'; '+signed(b['ci95'][1],6)+
          ']. Интервал включает ноль; две недельные разности отрицательны. ')
        if self.serving:
            cases=self.serving['cases']
            abstract+=('Отдельный warm-state benchmark 128 пользователей измеряет '+
              num(cases['manual']['full_warm_request']['mean_seconds'],3)+'/'+
              num(cases['auto_selected']['full_warm_request']['mean_seconds'],3)+
              ' с на запрос manual/выбранных окон при трёх seed-моделях в обеих ветвях. ')
        abstract+=('Представления не заменили сильную ручную базу; инвариантность якорей сама по себе '
          'не обеспечила устойчивого прироста. Выводы ограничены наблюдаемыми откликами, fixed candidate '
          'retrieval и заданными периодами; онлайн-эффект и универсальное превосходство не заявляются.')
        self.p(abstract,'abstract')
        self.p('<b>Ключевые слова:</b> рекомендательные системы; временные признаки; random search; KION; Yambda; стоимость inference.','small')
        self.p('<b>Отдельный исторический результат KION.</b> Исходная смесь шести моделей против трёх у manual: '
          f'ΔNDCG {signed(delta,6)}{interval}. Интервал включает ноль; дополнительная стоимость не превращается '
          'в установленную пользу, и после теста смесь не пересматривается.','small')
        self.h('1. Исследовательский вопрос')
        self.p('Промышленный рекомендатель редко выбирает между статистикой и машинным обучением как между '
          'двумя независимыми подходами. Часто несколько моделей формируют кандидатов, а бустинг объединяет '
          'их оценки с временными, категориальными и пользовательскими признаками. Поэтому полезный вопрос '
          'состоит в том, какие части такой системы можно автоматизировать без потери качества и какие '
          'дополнительные ресурсы для этого требуются.')
        self.p('Мы разделяем три гипотезы: дополнительные временные окна могут улучшить сильную базу; '
          'компактные признаки представлений могут заменить часть ручных агрегатов; геометрия относительно '
          'фиксированных объектов может дополнить скалярную близость latent-факторов. Эти гипотезы неэквивалентны: '
          'успешное дополнение ручной базы само по себе не означает успешной замены, а улучшение выбранной '
          'смеси не приписывается одному признаку. Контроль случайного поиска проверяет полезность '
          'importance-эвристики при ограниченном бюджете, а Yambda - перенос этой постановки на другой '
          'домен и несколько периодов, без переноса movie-specific длительности.')
        self.page()
    def related(self):
        self.h('2. Связь с опубликованными решениями')
        self.p('Многогенераторная схема опирается на открытый разбор решения Т-Банка для Rekko [1]: '
          'collaborative retrieval и его score/rank используются на втором уровне. Перенесён архитектурный '
          'приём, а не весь конкурсный ансамбль. Настройки WARP и BM25 адаптированы к данным и ограничениям '
          'эксперимента; опубликованный конкурсный результат не служит оценкой нашей системы.')
        self.p('Описание KION [2] важно не только как источник benchmark, но и как определение наблюдений: '
          'строка пары содержит последнюю дату и накопленные характеристики просмотра. Из решения '
          'Д. Тихонович [3] перенесены CatBoost-классификация, статистики дат и тренд после сглаживания Hamming7. '
          'Из открытого решения А. Петрова [4] использованы идеи демографической популярности, последних объектов '
          'и связей между историей и кандидатом. Наши временные срезы, широкий union и сравнение семейств '
          'зафиксированы отдельно, поэтому работа не заявляет полного воспроизведения победителей.')
        self.p('RecTools [5] задаёт конкретный weighted iALS-рецепт: фильтр длительности, веса confidence '
          'и совместное использование метаданных. В системе присутствуют также WARP/LightFM [6] и EASE [7]. '
          'EASE ограничен наиболее популярной частью prefix-каталога: это осознанная ресурсная адаптация '
          'линейного метода, а не обучение на полном каталоге.')
        self.p('FeatAug [8] рассматривает predicate-aware агрегаты по отношениям one-to-many и поиск запросов. '
          'У нас временной интервал играет роль предиката, а связь с категорией кандидата - роль условной '
          'агрегации. Однако SQL-движок, Bayesian optimization/TPE и beam search FeatAug не реализованы. '
          'Используется небольшая заранее материализованная грамматика и перебор префиксов групп. '
          'Featuretools [9] служит ориентиром для cutoff-aware определения признаков, но DFS также не запускается.')
        self.p('ProtoMF [10] мотивирует описание предпочтений через опорные направления. Наши anchors - '
          'реальные объекты, выбранные после обучения ALS, а не совместно обучаемые прототипы; prototype '
          'objective не воспроизводится. Это также не Nyström-аппроксимация: отсутствуют whitening и '
          'восстановление полного ядра. Идея ограничивается удобным для дерева представлением геометрии.')
        self.sub('Что не проверяется')
        self.p('Современные последовательные и экосистемные представления, включая Perseus, требуют другого '
          'набора событий и постановки. Здесь не обучались Transformer/SASRec и текстовый item encoder. '
          'Неполная последовательность последних взаимодействий KION не подменяет полный журнал просмотров. '
          'Поэтому результаты ALS-признаков не являются проверкой всех методов representation learning.')
        self.p('Практический вклад работы - согласованное сравнение этих ограниченных конструкций на общей '
          'выдаче кандидатов с прозрачной ценой в признаках, деревьях и числе моделей. Конкурентоспособность '
          'оценивается по собственным данным; авторитет исходного рецепта не трактуется как доказательство прироста.')
        self.page()
    def dataset(self):
        self.h('3. Данные, наблюдения и временной протокол')
        self.p('Используется KION: 5 476 251 строка, 962 179 пользователей и 15 706 объектов во взаимодействиях '
          '(15 963 в таблице каталога); даты в использованной выгрузке '
          'с 13 марта по 22 августа 2021 года. Наблюдение представляет уникальную пару user-item. '
          'Показов, достоверных отрицательных откликов и полной последовательности повторных просмотров нет. '
          'Релевантность в основном эксперименте бинарна и не зависит от будущей длительности целевой строки.')
        self.eq('H(t) = { (u,i,τ,…) : τ &lt; t };    Y<sub>u,t</sub> = { i : t ≤ τ<sub>ui</sub> &lt; t + 7 суток }.')
        self.p('Все признаки и генераторы используют строгий прошлый префикс H(t). Доступный каталог содержит '
          'объекты, встречавшиеся до t; seen-пары пользователя исключаются. Положительные объекты вне каталога '
          'или retrieval union не удаляются из знаменателей метрик. Таким образом, ranker не получает '
          'принудительно добавленных правильных ответов.')
        protocol=self.data['protocol'];sizes=protocol.get('cohort_sizes',{})
        rows=[['Обучение','19, 26 июля','26 июля, 2 августа',sizes.get('train',100000)],
              ['Validation','2, 9 августа','9, 16 августа',sizes.get('validation',15000)],
              ['Confirmation','16 августа','23 августа',sizes.get('test',40000)]]
        self.table(['Роль','Начало прогноза','Конец интервала','Пользователей'],rows,[.21,.27,.30,.22],
          'Таблица 1. Семидневные интервалы 2021 года и размеры случайных когорт.')
        self.p('Пользовательские когорты попарно не пересекаются; выбор выполняется без фильтра по будущей '
          'активности, seed 736291. Истории сохраняются целиком. Глобальные collaborative-модели и агрегаты '
          'используют весь доступный префикс, включая прошлые истории итоговых пользователей. Это '
          'трансдуктивный протокол: отсутствие пользователя в supervised-fit второго уровня не означает '
          'отсутствия его прошлого в retrieval.')
        self.p('Из итоговой когорты исключены 10 992 пользователя, чьи результаты просматривались в ранних '
          'экспериментах. При этом сама календарная неделя 16-23 августа уже исследовалась. Мы называем '
          'результат подтверждением на новых пользователях знакомого периода, а не полностью нетронутым '
          'временным holdout. Общие календарные тренды и каталог создают зависимость между пользователями.')
        self.p('Основная оценка условна на наличии прошлой истории и хотя бы одного будущего positive. '
          'Будущая активность задаёт только множество оцениваемых запросов; она не обучает генераторы '
          'и не расширяет каталог. Отдельно приводится оценка всех будущих активных пользователей '
          'с единым popularity fallback для не имеющих истории. Отсутствующие будущие positives не позволяют '
          'определить обычный пользовательский Recall; удержание всей исходной когорты здесь не измеряется.')
        self.page()
    def retrieval(self):
        self.h('4. Общие кандидаты и обучающая выборка')
        self.eq('C<sub>u,t</sub> = Union<sub>g</sub> TopK<sub>g</sub>(u,H(t)), без Seen<sub>u,t</sub>.')
        self.eq('Recall<sub>u</sub>@10 = |Top10<sub>u</sub> ∩ Y<sub>u</sub>| / |Y<sub>u</sub>|;'
          '<br/>NDCG<sub>u</sub>@10 = [Σ<sub>k=1,…,10</sub> rel<sub>u,k</sub>/log<sub>2</sub>(k+1)] / IDCG<sub>u</sub>@10.')
        self.p('В ideal DCG положительны первые min(10, |Y<sub>u</sub>|) позиций. Все пользовательские '
          'метрики усредняются с равными весами; missing positives уменьшают достижимую оценку.','small')
        self.p('Каждая feature-ветвь получает одинаковые user-item строки и одинаковый каталог. '
          'Бюджеты источников фиксируются до оценки ранжировщиков. Перекрывающиеся выдачи объединяются '
          'без дубликатов. Отсутствие кандидата является ошибкой retrieval; последующее ранжирование '
          'не может её исправить. Наряду с предметными признаками передаются scores и ranks генераторов.')
        self.table(['Источник','Бюджет на пользователя','Основная настройка'],[
          ['BM25; RP3beta','100; 100','top-K 200; RP3 α=1, β=0.6'],
          ['WARP; EASE; popularity','500; 500; 500','WARP 128D/20 эпох; EASE λ=1000'],
          ['Sex/age/income/kids popularity','По 60','Условная популярность'],
          ['Binary bias/hybrid ALS','По 50','Отдельные источники'],
          ['Weighted bias/hybrid ALS','По 100','bias128 и metadata32'],
          ['ALS96','0','Оценки доступны как признаки']], [.34,.25,.41],
          'Таблица 2. Общий многогенераторный union. Нулевой budget ALS96 не означает отсутствие его признаков.')
        self.p('EASE обучается на 5000 наиболее популярных объектов префикса, а не на полном каталоге. '
          'Это ограничивает охват его собственного источника, но не удаляет объекты из остальных источников '
          'или из истинных ответов. Широкий union включает и глобально популярные, и персональные объекты.')
        self.sub('Сэмплирование для второго уровня')
        self.eq('n<sub>neg</sub>(q) = min(3 n<sub>pos</sub>(q), 25, n<sub>available</sub>(q)).')
        self.p('Для каждого train-запроса сохраняются все найденные positives и случайные negatives '
          'без возвращения по указанному правилу. Запрос без найденного positive пропускается только '
          'при обучении второго уровня. CatBoost и LightGBM используют один и тот же sampled pool. '
          'Негативы выбираются непосредственно из полного union; отдельного предшествующего hard-negative '
          'поднабора нет. На validation и confirmation ранжируются полные candidate lists.')
        self.p('Ненаблюдаемая пара не означает доказанное отсутствие интереса: пользователь мог не видеть '
          'объект. Negative sampling и class weights меняют распределение обучающих примеров, поэтому '
          'выход классификатора используется только как ranking score. Ни калиброванная вероятность '
          'клика, ни catalogue-wide LogLoss не заявляются. После выбора рецепта к train добавляются '
          'сэмплированные validation-запросы; недельные query IDs остаются разными.')
        self.p('Сохранённый анализ validation union даёт около 933.7 кандидата на пользователя и macro '
          'candidate recall 0.8004 при равном весе недель. Oracle NDCG@10 равен примерно 0.8305: '
          'это недостижимая без меток сортировка найденных positives первыми, а не качество обученной модели. '
          'Удаление источника из union измеряет потерю покрытия, но не эффект переобучения ranker.')
        self.page()
    def windows(self):
        self.h('5. Ручные и автоматически сгенерированные признаки')
        self.p('Manual является сильной базой: категории пользователя и объекта, item identity, scores/ranks '
          'общих генераторов, характеристики последних пяти объектов и сходства с ними, ручные временные '
          'агрегаты и свойства завершённых пар. Добавляются median/std дат, сглаженный Hamming7-тренд, '
          'доли наиболее недавних строк и частоты категорий prefix-каталога. Это промышленно мотивированная '
          'архитектура второго уровня, но не утверждение о готовности всей системы к production SLA.')
        self.eq('N<sub>u,w</sub>(t) = Σ<sub>e in H(t)</sub> 1[u<sub>e</sub>=u] 1[t-w ≤ τ<sub>e</sub> &lt; t],'
          '<br/>Share<sub>u,w</sub> = N<sub>u,w</sub> / max(N<sub>u,all</sub>,1).')
        self.p('Аналогично считаются item counts и shares. Ручная база использует окна 1/7/30 дней '
          'и всю историю. Дополнительная грамматика материализует окна 2/3/5/10/14/21/45/60/90 дней. '
          'Единица наблюдения здесь - сохранённая пара с последней датой, а не каждый просмотр фильма. '
          'Такое различие особенно важно для интерпретации коротких окон.')
        self.p('Пусть X<super>(w)</super> - sparse user-item матрица прошлого окна, A<super>r</super> - '
          'бинарная item-token матрица отношения r, например жанра или страны. Профиль и affinity задаются как')
        self.eq('P<super>r,w</super> = X<super>(w)</super>A<super>r</super>;'
          '<br/>Affinity<sub>u,i</sub><super>r,w</super> = P<sub>u</sub><super>r,w</super>(A<sub>i</sub><super>r</super>)<super>T</super>.')
        self.p('Нормированная affinity делится на max(сумма token-counts профиля,1). Совпадение нескольких '
          'категорий даёт несколько вкладов: это не просто индикатор наличия хотя бы одного общего жанра. '
          'Отношения и операторы заранее заданы; алгоритм не изобретает произвольные формулы или связи.')
        self.sub('Характеристики завершённых пар')
        self.p('Для last_watch_dt &lt; cutoff все просмотры, вошедшие в сохранённую строку, лежат '
          'в прошлом. Поэтому исторические total_dur и watched_pct допустимы при принятой семантике '
          'дат benchmark. Count/share qualified-пар использует total_dur ≥ 300; также считаются средние '
          'log(1+duration), pct и доля сильных пар с pct &gt; 10. Длительность строки, выбранной по недавней '
          'последней дате, не является длительностью только внутри окна. Будущую строку нельзя частично '
          'восстановить для прошлого.')
        self.sub('Групповой выбор окон')
        self.p('Train-fitted CatBoost выбранной development-конфигурации задаёт importance-порядок групп. '
          'Затем validation выбирает лучший непустой префикс длиной 1/2/3/5; manual сохраняется всегда. '
          'В случае равенства остаётся первый, меньший префикс. Это ограниченная автоматизация выбора '
          'окон, а не минимизация числа признаков и не общий AutoML по всем формулам. Отдельно проверяется '
          'ветвь со всеми окнами, чтобы отбор не подменял сравнение полным набором.')
        self.page()
    def representations(self):
        self.h('6. Представления и геометрия относительно якорей')
        self.p('Взвешенный ALS применяет строгий временной префикс, затем оставляет строки с duration ≥ 300. '
          'Вес наблюдаемой пары равен 3 при pct &gt; 10 и 1 иначе; confidence - α×weight, α=10, '
          'а для пропусков confidence равен 1. Регуляризация .5, десять итераций. Сначала модель обучается '
          'на активной подматрице; факторы разворачиваются к исходным ID. Пользователь без qualified history '
          'получает popularity fallback weighted-префикса, неактивный item - нулевые факторы.')
        self.p('Совместный metadata32-режим использует опубликованный solve-then-restore приём RecTools: '
          'свободные и фиксированные координаты решаются совместно, затем metadata-блок восстанавливается. '
          'Это не точная constrained-минимизация. Отдельный bias128-режим добавляет paired-факторы '
          'к константам и даёт latent dot плюс user/item bias. Raw labels, seen и каталог при этом '
          'не фильтруются по длительности: фильтр меняет источник сигнала, а не задачу оценки.')
        self.p('Произвольные координаты факторов могут вращаться между prefix-fit. Вместо них базовые '
          'representation-признаки описывают dot/cosine/norm, скалярные вклады фиксированных semantic '
          'metadata-блоков и active-флаги. Ветка representations сохраняет metadata, часть retrieval '
          'и history similarity; она заменяет ручные агрегаты, но не является чистой моделью эмбеддингов.')
        self.sub('Фиксированные item anchors')
        self.p('На первом train-префиксе 19 июля выбираются до 16 активных ненулевых item-факторов. '
          'Pool - первые 1000 по 30-дневной raw-популярности; равенства разрешаются ID. Первый anchor '
          'самый популярный, следующие максимизируют минимальное cosine-расстояние до выбранных. '
          'Поздние префиксы используют те же ID и свои текущие факторы без повторного отбора.')
        self.eq('d<sub>j</sub> = u a<sub>j</sub><super>T</super>;    c<sub>j</sub><super>u</super> = cos(u,a<sub>j</sub>);'
          '<br/>c<sub>j</sub><super>i</super> = cos(v<sub>i</sub>,a<sub>j</sub>);    p<sub>j</sub> = c<sub>j</sub><super>u</super>c<sub>j</sub><super>i</super>.')
        self.p('Дополнительный флаг задаёт доступность anchor. При отсутствии или нулевом знаменателе '
          'соответствующие величины обнуляются. В joint ALS геометрия использует только свободный latent-блок, '
          'без фиксированных metadata. При общем ортогональном Q: (uQ)(aQ)<super>T</super> = '
          'uQQ<super>T</super>a<super>T</super> = ua<super>T</super>, а нормы сохраняются. Следовательно, '
          'признаки инвариантны к совместному ортогональному вращению одной модели.')
        self.p('Это не выравнивает независимо обученные prefix-пространства. Шестнадцать направлений '
          'в 32D оставляют nullspace размерности 16. Train-диагностика даёт effective rank около 14.585 '
          'и 21.88% остаточной энергии raw active item-факторов после проекции на anchor span. '
          'Эти числа описывают геометрию, а не качество; 80 дополнительных колонок могут быть избыточны.')
        self.page()
    def learning(self):
        self.h('7. Обучение, бюджеты и сравниваемые системы')
        self.p('Четыре основные ветви manual/auto_windows/auto_full/representations получают две '
          'конфигурации CatBoost и две LightGBM/LambdaRank. CatBoost: depth/L2 = 9/27 либо 7/15, '
          'learning rate .03, subsample .97, scale_pos_weight 3, до 2000 деревьев. Каждые 50 '
          'деревьев оценивается полный NDCG и выбирается лучший prefix непрерывного обучения. '
          'LightGBM проверяет leaves/L2/min_child = 15/.1/80 либо 31/5/100, до 1200 деревьев, '
          'learning rate .035, early stopping 100 по той же полной метрике.')
        self.p('Критерий выбора - равное среднее недельных NDCG, а не объединение всех строк двух недель. '
          'Это предотвращает доминирование более активного периода, но не создаёт независимой оценки: '
          'те же недели используются для HPO, importance-порядка, числа деревьев, feature-prefix и смеси. '
          'Development point estimates поэтому имеют оптимизм выбора.')
        rows=[]
        for name in ORDER:
            if name not in self.choices:continue
            c=self.choices[name];rows.append([NAMES[name],c['family'],c.get('features',len(c.get('columns',[]))),c['trees'],str(c['params'])])
        self.table(['Ветка','Семейство','Признаки','Деревья','Выбранные параметры'],rows,[.27,.15,.12,.12,.34],
          'Таблица 3. Выбор на development. Указано число деревьев одной seed-модели.')
        self.p('Бюджеты не полностью одинаковы. Auto_selected перебирает четыре префикса только на CatBoost '
          'с настройками, выбранными для auto_windows. Две новые anchor-ветви получают по две прежние '
          'CB-конфигурации; отдельной LightGBM-сетки у них нет. Следовательно, best-to-best сравнение '
          'характеризует ограниченную процедуру разработки, а не только причинный эффект feature family.')
        self.sub('Refit и итоговая смесь')
        self.p('Development использует seed 17. После выбора все варианты переобучаются на train '
          'и сэмплированной validation с фиксированным числом деревьев, seeds 17/42/73. Raw scores '
          'сначала усредняются внутри каждой ветви. Затем rank-mixture использует ранги полного '
          'user candidate list с разрешением score-равенств по item ID.')
        self.eq('f̄<sub>b</sub>(u,i) = (1/3) Σ<sub>s in {17,42,73}</sub> f<sub>b,s</sub>(u,i);'
          '<br/>S(u,i) = 0.5/(60+r<sub>windows</sub>) + 0.5/(60+r<sub>selected</sub>).')
        self.p('Выбранная смесь содержит шесть CatBoost-моделей, manual - три. Это не среднее шести '
          'индивидуальных рангов и не равный model-count эксперимент. Три seeds не являются тремя '
          'независимыми итоговыми выборками. Все семь диагностических вариантов вместе содержат 21 '
          'модель; стоимость их общего scoring нельзя выдавать за latency одной выбранной системы.')
        counts=self.data.get('row_counts',{})
        self.p('Материализованный train-pool содержит '+str(counts.get('sampled_training_rows','не записано'))+
          ' пар, полная validation - '+str(counts.get('validation_candidate_rows','не записано'))+
          '; после добавления сэмплированной validation финальный refit использует '+
          str(counts.get('final_training_rows','не записано'))+' обучающих пар. Число строк не равно размеру пользовательской когорты.','small')
        self.page()
    def development(self):
        self.h('8. Development: отбор окон и anchor-диагностика')
        rows=[]
        for name in ORDER:
            if name in self.val and name in self.choices:
                m=self.val[name];rows.append([NAMES[name],num(m['ndcg@10'],6),num(m['recall@10'],6)])
        self.table(['Ветка','Средний NDCG@10','Средний Recall@10'],rows,[.5,.25,.25],
          'Таблица 4. Равные веса двух validation-недель. Это метрики после подбора, seed 17.')
        selected=self.choices.get('auto_selected',{}).get('selected_window_groups',[])
        self.p('Выбранный префикс дополнительных окон: '+escape(', '.join(selected) or 'см. сохранённую конфигурацию')+'. '
          'В этой задаче он дополняет manual. Отбор показывает полезность конкретной ограниченной '
          'грамматики и двух development-периодов; он не доказывает оптимальность этих окон для других недель.')
        matched=self.data.get('matched_catboost_comparisons',[])
        if matched:
            rows=[]
            for m in matched:
                d=m['delta_anchor_minus_parent'];params=m['params']
                rows.append([NAMES.get(m['parent_variant'],m['parent_variant']),str(params),
                  str(m['parent_trees'])+' → '+str(m['anchor_trees']),signed(d['ndcg@10'],6),signed(d['recall@10'],6)])
            self.table(['База + anchors','Параметры CB','Деревья','ΔNDCG','ΔRecall'],rows,[.25,.27,.18,.15,.15],
              'Таблица 5. Matched trials: одинаковые family/params/seed; delta = anchor - parent.')
            negative=sum(m['delta_anchor_minus_parent']['ndcg@10']<0 for m in matched)
            self.p(f'Из {len(matched)} matched-проб отрицательная NDCG-разность получена в {negative}. '
              'Число деревьев каждая сторона выбирала независимо на validation. Даже совпадение '
              'выбранных tree counts в отдельной паре не делает результат независимой повторной '
              'проверкой: параметры и точка сравнения рассматриваются после разработки.')
        weekly=[a['ndcg@10']-m['ndcg@10'] for a,m in zip(self.val['auto_selected']['weeks'],self.val['manual']['weeks'])]
        self.p('Для auto_selected недельные дельты важнее одного среднего. В сохранённом development '
          'NDCG-разность относительно manual равна '+signed(weekly[0],6)+' на первой неделе и '+signed(weekly[1],6)+
          ' на второй. Положительное среднее не означает стабильного улучшения в каждом периоде. '
          'Финальная смесь не включает anchor-ветви; это результат выбора на validation, '
          'а не основание убрать их отрицательные наблюдения из исследования.')
        self.p('Ниже итоговые метрики оценивают refit и ансамбли, а не те же single-seed модели. '
          'Изменение development → confirmation отражает одновременно новые запросы, календарный '
          'период, дополнительное обучение на validation и усреднение seeds; интерпретация только '
          'как temporal drift была бы некорректна.')
        self.page()
    def final_results(self):
        self.h('9. Итоговая оценка на новых пользователях')
        rows=[]
        for name in ORDER:
            if name not in self.test:continue
            m=self.test[name];rows.append([NAMES[name],num(m['ndcg@10'],6),num(m['recall@10'],6),
                num(m.get('map@10'),6),str(m.get('topk_item_coverage',m.get('coverage@10','-')))])
        m=self.test['manual'];count=self.data.get('row_counts',{}).get('test_candidate_rows','не записано')
        self.table(['Система','NDCG@10','Recall@10','MAP@10','Объектов в top10'],rows,[.34,.18,.18,.18,.12],
          f"Таблица 6. Confirmation: {m['users']} warm users, {count} candidate pairs, общий candidate recall {num(m.get('candidate_recall'),6)}. Последняя колонка - число различных рекомендованных объектов.")
        self.fig('quality.png',220,'Рисунок 1. Development и confirmation показаны отдельно. Соединять эти точки как оценку одного и того же fit некорректно.')
        self.p('Пропущенные positives, включая cold items, остаются в знаменателях. Основные результаты '
          'условны на future-positive warm users, поэтому их нельзя переносить на всех зарегистрированных '
          'пользователей. Популярность и generator RRF оцениваются на тех же запросах и кандидатах; '
          'они отделяют пользу сложного ranking от самого качества общего retrieval.')
        self.p('NDCG и Recall могут дать разные направления изменения: первый показатель чувствителен '
          'к ранним позициям и индивидуальному ideal DCG, второй считает долю всех истинных ответов '
          'среди десяти рекомендаций. Ни один из них не измеряет длительность просмотра, разнообразие '
          'потребления или причинное удовлетворение пользователя.')
        self.page()
    def random_control(self):
        self.h('Равный бюджет: importance и случайный выбор окон')
        d=self.random;design=d['design'];importance=d['importance'];random=d['random']
        self.p('Исходный поиск признаков не сравнивался со случайным поиском при том же бюджете. '
          'Для устранения этого пробела после исходной итоговой оценки выполнен отдельный '
          'development-контроль. До первого нового обучения зафиксированы четыре независимых равномерных '
          'набора из девяти групп, размеры 1/2/3/5, RNG seed 20261025. Выбор без возвращения внутри '
          'набора не запрещает совпадения между пробами. Первый набор случайно совпал с importance-k1; '
          'он не перерисовывался, а его повторное обучение точно воспроизвело метрики и число деревьев.')
        self.p('Обе процедуры сохраняют 181 manual-признак и получают CatBoost depth=7, L2=15, seed17, '
          '2000 непрерывно обученных деревьев. Полный validation NDCG оценивается каждые 50 деревьев; '
          'выбираются лучший префикс деревьев и лучший из четырёх наборов по равному среднему двух недель. '
          'Сохранённая обучающая выборка, полные validation-кандидаты и знаменатели совпадают; новых negatives '
          'и правильных кандидатов не добавляли. Importance-результаты взяты из исходных четырёх обучений.')
        rows=[]
        for i,r in zip(importance['trials'],random['trials']):
            windows=lambda x:','.join(g.removeprefix('window_') for g in x['groups'])
            rows.append([i['cardinality'],windows(i),windows(r),num(i['metrics']['ndcg@10'],6),
                         num(r['metrics']['ndcg@10'],6),signed(r['metrics']['ndcg@10']-i['metrics']['ndcg@10'],6)])
        self.table(['k','Importance: дни','Random: дни','NDCG I','NDCG R','R - I'],rows,
          [.06,.23,.25,.15,.15,.16], 'Development-контроль: четыре обучения на наборах окон для каждой процедуры; не новая итоговая выборка.')
        shared=design['shared_precompute']['fit_seconds']
        self.p('Два исходных обучения CB со всеми окнами, из которых выбраны общие параметры и '
          'importance-порядок, стоят '+num(shared,2)+' с и являются общим precompute. '
          'Четыре обучения на наборах окон стоят '+num(importance['total_fit_seconds'],2)+' с у importance и '+
          num(random['total_fit_seconds'],2)+' с у random; вместе с общим precompute - '+
          num(shared+importance['total_fit_seconds'],2)+' и '+num(shared+random['total_fit_seconds'],2)+
          ' с. Равенство числа обучений и лимита деревьев не означает равенства затраченного времени.','small')
        self.fig('random_control.png',145,'Один заранее заданный random-search probe. Точки выбраны по tree-prefix на тех же development-неделях; интервалы не оценивались.')
        i,r=importance['best'],random['best'];delta=d['comparison']
        self.p('Лучший random-набор [3,21] даёт '+num(r['metrics']['ndcg@10'],6)+
          ' против '+num(i['metrics']['ndcg@10'],6)+' у importance [3,2,5]. Разности random - importance '
          'по неделям: '+', '.join(signed(v,6) for v in delta['random_minus_importance_week_ndcg@10'])+
          '. Это одна реализация ограниченного поиска, а не оценка его дисперсии или формальная '
          'проверка превосходства. В random-k5 порядок блоков колонок отличается от importance-сериализации '
          'и может влиять на fixed-seed CB. Наборы и бюджет после результатов не расширялись; '
          'новое итоговое обучение random на ранее открытом KION-тесте не выполнялось.','small')
        self.page()
    def uncertainty(self):
        self.h('10. Парные различия и статистическая интерпретация')
        self.p('Для пользователя u определяется d<sub>u</sub> = metric<sub>challenger,u</sub> - '
          'metric<sub>reference,u</sub>. Bootstrap ресэмплирует пользователей целиком с возвращением '
          'и усредняет d; из 2000 повторов берутся percentile-границы 2.5% и 97.5%, seed 2026. '
          'Это парная процедура: обе системы оцениваются на одних и тех же user queries.')
        rows=[]
        for name,by in self.paired.items():
            label='Ручная база - popularity' if name=='manual_vs_popularity' else NAMES.get(name,name)+' - ручная база'
            rows.append([label,ci(by['ndcg']),ci(by['recall'])])
        self.table(['Сравнение','ΔNDCG [95% интервал]','ΔRecall [95% интервал]'],rows,[.36,.32,.32],
          'Таблица 7. Основные объявленные сравнения. Дельты в единицах метрик, не в процентах.')
        if self.diag:
            rows=[]
            for d in self.diag.values():
                label=NAMES.get(d['challenger'],d['challenger'])+' - '+NAMES.get(d['reference'],d['reference'])
                rows.append([label,ci(d['bootstrap']['ndcg']),ci(d['bootstrap']['recall'])])
            self.table(['Диагностическая пара','ΔNDCG [95% интервал]','ΔRecall [95% интервал]'],rows,[.36,.32,.32],
              'Таблица 8. Дополнительные пары определены до итоговой оценки и не участвуют в выборе primary.')
        self.fig('paired_deltas.png',170,'Рисунок 2. Сохранённые парные 95% интервалы; горизонтальная шкала в процентных пунктах. Интервалы маргинальные, без multiplicity correction.')
        if 'primary' in self.paired:
            d=self.paired['primary']['ndcg']
            self.p('Для primary относительно manual '+change_text(d)+'. '
              'Это описательная оценка фиксированных систем на данной пользовательской выборке. '
              'Интервал, пересекающий ноль, не доказывает равенство или отсутствие практически значимого '
              'ущерба. Граница non-inferiority заранее не задавалась.')
        self.p('Bootstrap не переобучает модели и не меняет неделю, каталог или retrieval. Он не учитывает '
          'вариативность HPO и будущих периодов. Много ветвей, метрик и диагностических пар увеличивают '
          'риск выборочного прочтения результатов; здесь не проводится поправка на множественность '
          'и не объявляется набор независимых confirmatory-тестов.')
        self.page()
    def costs(self):
        self.h('11. Ресурсы, компактность и cold fallback')
        cost=self.data.get('costs',{});models=cost.get('models',{})
        rows=[]
        for name in ORDER:
            if name not in models:continue
            m=models[name];size=m.get('model_bytes_all_seeds',m.get('model_bytes'))
            seconds=m.get('fit_seconds_all_seeds',m.get('fit_seconds'))
            rows.append([NAMES[name],str(m.get('seeds',3)),num(seconds,1),num(size/1e6 if size is not None else None,1)])
        if rows:self.table(['Ветка','Seeds','Refit, с','Модели, MB'],rows,[.4,.12,.24,.24],
          'Таблица 9. Суммарная стоимость фиксированного refit всех seeds каждой ветви, без общего retrieval.')
        self.p('Development search V4 занял '+num(cost.get('search_seconds'),1)+' с; дополнительный поиск '
          'anchor-ветвей - '+num(cost.get('extension_search_seconds'),1)+' с. Полный итоговый scoring всех '
          'диагностических ветвей с загрузкой моделей и projected I/O - '+
          num(cost.get('prediction_wall_seconds_with_load_io'),1)+' с; собственно преобразование '
          'признаков и native prediction - '+num(cost.get('prediction_seconds_all_models'),1)+' с. '
          'Эти стадии указаны раздельно; отсутствующие замеры не заменяются нулём.')
        rss_dev=cost.get('development_peak_resident_bytes');rss_final=cost.get('final_peak_resident_bytes')
        self.p('Измеренный peak RSS development/final составляет '+
          num(rss_dev/1e9 if rss_dev is not None else None,2)+' / '+
          num(rss_final/1e9 if rss_final is not None else None,2)+
          ' GB. Это пики отдельных процессов, а не сумма и не изолированный serving benchmark.','small')
        self.p('Расходы общих CF-моделей и построения snapshots разделяются между ветвями. Исходное '
          'fit-время кешированного генератора не является новым wall time каждого использования. '
          'Время snapshot включает retrieval, построение признаков, I/O и возможный fit/чтение кеша; '
          'ниже отдельный benchmark выделяет feature-only и warm-request стадии. Поэтому суммы различных '
          'сохранённых длительностей не интерпретируются как точное end-to-end время.')
        def sum_cost(names,key):
            values=[models.get(name,{}).get(key) for name in names]
            return sum(values) if values and all(value is not None for value in values) else None
        used=[name for name,weight in self.mix.items() if weight>0 and name in self.choices]
        serving=[]
        for label,names in [('Ручная база',['manual']),('Выбранная смесь',used)]:
            count=sum(len(self.choices[name].get('seeds',[17,42,73])) for name in names)
            size=sum_cost(names,'model_bytes_all_seeds')
            serving.append([label,count,num(sum_cost(names,'fit_seconds_all_seeds'),1),num(size/1e6 if size is not None else None,1)])
        self.table(['Система','GBDT-моделей','Refit, с','Модели, MB'],serving,[.4,.18,.21,.21],
          'Таблица 10. Ресурсы исходной смеси и базы. Общие retrieval-модели исключены; время запроса измерено отдельно.')
        full=self.data.get('all_future_active_user_metrics',self.data.get('test',{}).get('all_future_active_user_metrics',{}))
        if full:
            rows=[]
            for name in ['manual','auto_selected','representations','primary','popularity']:
                if name in full:
                    m=full[name];rows.append([NAMES[name],str(m['users']),num(m['ndcg@10'],6),num(m['recall@10'],6)])
            cold=self.data.get('cold_user_fallback',{}).get('users','не записано')
            if rows:self.table(['Система','Warm + cold users','NDCG@10','Recall@10'],rows,[.4,.2,.2,.2],
              f'Таблица 11. Все будущие активные пользователи; {cold} cold users получают общий fallback.')
        self.p('Warm+cold и warm оценивают разные наборы запросов; более высокое среднее не означает '
          'улучшения на прежних warm users. Cold-only запрос не требует GBDT. Само уменьшение входной '
          'размерности ещё не доказывает ускорения: важно исключить построение неиспользуемых блоков '
          'и сравнивать одинаковые границы измерения. Именно это делает следующий benchmark.')
        self.page()
    def serving_results(self):
        self.h('Измерение стоимости запроса')
        d=self.serving;cases=d['cases'];startup=d['shared_startup']
        self.p('Из prefix-пользователей равномерно выбраны 128 ID без фильтра по будущей активности '
          '(seed2026). Все три системы получают этих же пользователей и '+str(d['candidate_rows'])+
          ' candidate pairs. Prefix-fit и GBDT не переобучаются. Один warmup и три измеряемых повтора '
          'выполняются с четырьмя GBDT-потоками и одним BLAS-потоком. Величины ниже - секунды на '
          'batch из 128 пользователей; это не индивидуальная latency и не production SLA.')
        self.p('Selective store строит только требуемые окна и колонки: 181 у manual, 223 у выбранных '
          'окон и 307 у исходной смеси. Значения всех требуемых признаков точно совпали с полным '
          '398-колоночным store. Неиспользуемые latent/anchor-блоки не строятся. Feature-only '
          'измеряет build на готовых кандидатах; full warm request включает retrieval, selective build, '
          'prediction загруженных моделей и выдачу top10.')
        rows=[]
        for key in ['manual','auto_selected','primary']:
            c=cases[key]
            span=lambda name:num(c[name]['mean_seconds'],3)+' ['+num(c[name]['min_seconds'],3)+'; '+num(c[name]['max_seconds'],3)+']'
            rows.append([NAMES[key],c['required_feature_columns'],c['seed_models'],span('feature_only'),
                         span('full_warm_request'),num(c['prefix_aggregate_store_constructor_seconds'],3)])
        self.table(['Система','F','Моделей','Build, с','Warm request, с','Store startup, с'],rows,
          [.25,.06,.09,.20,.23,.17], 'Среднее [min;max] трёх повторов. Prefix aggregate-store constructor указан отдельно от запросов.')
        self.fig('serving_cost.png',165,'Одинаковый batch и selective feature construction. Штрихи показывают диапазон трёх локальных повторов, не доверительный интервал.')
        selected=cases['auto_selected']['full_warm_request']['mean_seconds']/cases['manual']['full_warm_request']['mean_seconds']
        mixture=cases['primary']['full_warm_request']['mean_seconds']/cases['manual']['full_warm_request']['mean_seconds']
        self.p('Выбранные окна требуют '+num(selected,3)+'× времени warm-запроса manual; шесть моделей '
          'исходной смеси - '+num(mixture,3)+'×. Это конкретный quality-cost результат этого стенда. '
          'Он не доказывает пропорциональной зависимости latency от числа признаков или моделей.')
        self.sub('Что вынесено за границу warm request')
        self.p('Общая загрузка prefix-index: '+num(startup['index_load_seconds'],3)+' с; дополнительных '
          'генераторов: '+num(startup['extra_generator_load_seconds'],3)+' с; чтение raw и prefix-фильтр: '+
          num(startup['raw_read_and_prefix_seconds'],3)+' с. Первичное построение кандидатов для '
          'feature-only замера: '+num(startup['initial_candidate_generation_seconds'],3)+' с. '
          'Aggregate-store startup занимает 6.54-7.28 с и может амортизироваться между запросами.','small')
        self.p('Первое / повторное чтение ансамбля в данном процессе: manual '+
          num(cases['manual']['first_ensemble_load_seconds'],3)+' / '+num(cases['manual']['repeat_ensemble_load_seconds'],3)+
          ' с; selected '+num(cases['auto_selected']['first_ensemble_load_seconds'],3)+' / '+
          num(cases['auto_selected']['repeat_ensemble_load_seconds'],3)+' с; смесь '+
          num(cases['primary']['first_ensemble_load_seconds'],3)+' / '+num(cases['primary']['repeat_ensemble_load_seconds'],3)+
          ' с. OS cache не сбрасывался: это наблюдаемое process-load I/O, а не физически холодный диск. '
          'Прогрев моделей, глобальный prefix-fit и запуск приложения не скрываются внутри warm-цифр.','small')
        self.page()
    def yambda_protocol(self):
        self.h('Внешняя проверка: Yambda и новые периоды')
        self.p('Yambda-5B - открытые музыкальные взаимодействия с разделением organic и '
          'recommendation-driven событий [11]. Для этой работы использована публичная sequential '
          '50M-выборка [12], а не весь многомиллиардный набор. В отличие от KION, сохранённые '
          'последовательности позволяют считать повторные прослушивания в каждом прошлом окне. '
          'Проверяется перенос идеи временной грамматики; официальный Yambda benchmark не воспроизводится.')
        self.p('До чтения новых Task1-целей зафиксированы 4000 пользователей вне Task2-когорты: '
          '2500 train, 500 validation, 1000 test. Исходные данные уже были известны по другой задаче; '
          'это не заявление о глобально невиденном raw-наборе. Выбранные ниже недели предшествуют '
          'периодам quality-оценки Task2, а его пользователи исключены. Когорты выбраны без требования '
          'будущей активности. Все raw listen histories выбранных пользователей сохраняются, включая '
          'organic, повторы и редкие треки вне рабочего каталога.')
        self.p('Для этих пользователей сохранены '+format(self.yambda['history_listens'],',').replace(',',' ')+
          ' raw listens без усечения длинных историй. В каждом срезе из них используется только строгий прошлый префикс.','small')
        self.eq('d = timestamp / 86400;    H(d) = {e: timestamp<sub>e</sub> &lt; 86400 d}.')
        self.table(['Роль','Начало прогноза, относительный день','Семидневные цели','Когорта'],[
          ['Train','133; 140','[133,140); [140,147)','2500'],
          ['Validation','147; 154','[147,154); [154,161)','500'],
          ['Final','161; 168; 175','[161,168); [168,175); [175,182)','1000']],
          [.17,.31,.36,.16], 'Предварительно заданные относительные интервалы. Дни не являются Unix- или календарными датами.')
        self.sub('Цель и доступный каталог')
        self.eq('Y<sub>u,d</sub> = unique { item<sub>e</sub>: d ≤ t<sub>e</sub>/86400 &lt; d+7,'
                ' event_type = listen, is_organic = 0 }.')
        self.p('Целью являются все записанные recommendation-driven listen-события, включая нулевой '
          'played ratio. Это наблюдаемый факт взаимодействия, не satisfaction и не официальный '
          'Listen+ с порогом свыше 50% [12]. Никакой KION-фильтр duration≥300 к музыке не переносится. '
          'Ранее прослушанный трек остаётся допустимой рекомендацией и будущим positive; seen exclusion '
          'здесь намеренно отсутствует.')
        self.p('Каталог - top10000 по числу прошлых listens всех выбранных пользователей строго до '
          'дня133, с разрешением равенств по track ID. Затем ID каталога фиксируются. Это трансдуктивное '
          'использование прошлых историй всех когорт, а не будущих меток. Истинные ответы вне каталога '
          'не удаляются из IDCG/Recall. Static artist mapping сводится к минимальному artist ID '
          'объекта; его историческая доступность неизвестна.')
        self.p('Главная популяция - все будущие активные пользователи, в том числе без истории, '
          'с единым popularity fallback. No-positive queries не получают искусственный нулевой NDCG: '
          'они учитываются в размерах когорт, но метрика интереса для них не определена. Warm-оценка '
          'показывается отдельно. Пользователь с rare-only прошлым остаётся warm, хотя CF-каталог '
          'может не содержать его истории.')
        self.page()
    def yambda_methods(self):
        self.h('Yambda: сильная база и симметричный поиск')
        self.p('Общий retrieval объединяет popularity500 за последние 30 дней, BM25 ItemKNN200 '
          '(K1=1.2, B=.8, topK=200) и ALS100 (32 фактора, α=10, regularization=.5, десять итераций). '
          'CF использует бинарный user-track префикс на фиксированном каталоге. Объединение не дополняется '
          'целевыми треками. Count-признаки сохраняют число повторных прослушиваний, '
          'включая события вне каталога там, где определяется история пользователя или artist.','small')
        self.table(['Ручная группа','Содержание'],[
          ['Retrieval и категории','Scores/ranks трёх источников, track/artist ID, последний raw track.'],
          ['История и recency','Давность пользователя/item/пары, span истории, число distinct raw tracks, cold flag.'],
          ['Окна 1/7/30/all','User/item/pair count и доли; artist affinity и её доля.'],
          ['Дополнительная грамматика','Те же 8 агрегатов в окнах 2/3/5/10/14/21/45/60/90.']],
          [.31,.69], 'Manual уже содержит статистики четырёх масштабов времени; automatic добавляет окна, а не создаёт отсутствующую базу.')
        self.p('Для обучения каждого query равномерно сохраняются максимум 20 найденных positives '
          'и min(3p,25,z) negatives без возвращения, где p - число сохранённых positives, '
          'z - число доступных истинных нулевых кандидатов. '
          'Остальные positives исключаются из обучения, а не превращаются в negatives. Все ветви получают '
          'ту же выборку; запросы без найденных positives пропускаются только при обучении. Validation и final '
          'используют полные общие candidate lists и полный истинный знаменатель.')
        self.sub('Бюджет и замораживание выбора')
        self.p('Параметры перенесены без нового HPO: CB depth7/L2=15, learning rate .03, seed17, '
          '2000 непрерывных деревьев, выбор prefix каждые50 по равному среднему двух недель. '
          'Общие обучения manual и всех окон дают baseline и importance-порядок обученной модели. Importance '
          'проверяет его префиксы 1/2/3/5, random - четыре заранее заданных независимых набора '
          'тех же размеров, seed20261026. В обеих ветвях выбранные блоки колонок сериализуются '
          'в одном каноническом порядке дней: различается состав признаков.')
        self.p('Оба автоматических варианта обязаны выбрать непустой набор k из {1,2,3,5}; '
          'manual не является их fallback. Это диагностическое сравнение заранее заданных '
          'процедур, а не правило, обязывающее применять окна при худшем validation-результате.','small')
        self.p('All-window модель сама выбирает tree-prefix на этих же двух validation-неделях. '
          'Поэтому полученный из её train-fitted importance порядок не является процедурой, '
          'полностью независимой от validation. Эта общая стадия и её стоимость указаны явно.','small')
        self.p('После development для каждого из трёх фиксированных вариантов manual/importance/random '
          'выбираются набор окон и число деревьев; затем все переобучаются с seeds17/42/73 на train '
          'и одинаково сэмплированной validation. На трёх final-неделях веса GBDT, наборы окон и деревья '
          'не меняются. Глобальный CF и признаки обновляются перед каждой неделей только по её прошлому. '
          'Победитель по final не выбирается.')
        self.sub('Заранее заданный основной контраст')
        self.eq('Δ = (1/3) Σ<sub>w in {161,168,175}</sub> [mean<sub>u in A<sub>w</sub></sub>NDCG<sub>importance,u,w</sub>'
                ' - mean<sub>u in A<sub>w</sub></sub>NDCG<sub>manual,u,w</sub>].')
        self.p('Основной контраст - importance minus manual на all-active; random-контрасты, Recall '
          'и отдельные недели вторичны. Среднее берётся по трём недельным macro, а не по доступным '
          'неделям каждого пользователя. Для неопределённости используется общий user-cluster '
          'bootstrap по active-user union; его единицы и границы приведены вместе с интервалами.')
        self.page()
    def yambda_results(self):
        self.h('Yambda: результаты по неделям')
        d=self.yambda
        names={'manual':'Manual','importance':'Importance','random':'Random'};rows=[]
        selected=d['selected']
        v={arm:selected[arm]['metrics']['ndcg@10'] for arm in names}
        text=('На development NDCG@10: manual '+num(v['manual'],6)+', importance '+
              num(v['importance'],6)+', random '+num(v['random'],6)+'. ')
        if v['manual']>max(v['importance'],v['random']):
            text+=('Практическое правило с возможностью пустого набора могло бы оставить manual уже '
                   'на validation. Здесь три варианта сохраняются ради диагностической проверки; '
                   'добавление окон вопреки этой оценке не рекомендуется.')
        else:text+='По итоговым неделям варианты не переключаются; сравниваются замороженные процедуры.'
        self.p(text)
        for week in d['test_weeks']:
            systems=week['systems'];a={arm:systems[arm]['all_active']['ndcg'] for arm in names}
            rows.append([week['day'],num(a['manual'],6),num(a['importance'],6),num(a['random'],6),
                         signed(a['importance']-a['manual'],6),signed(a['importance']-a['random'],6)])
        self.table(['День','Manual','Importance','Random','I - M','I - R'],rows,
          [.08,.18,.18,.18,.19,.19], 'Полные знаменатели future-positive; все активные пользователи и общий cold fallback. По три seed-модели на вариант.')
        pooled=d['pooled_equal_week']['all_active']
        self.table(['Система','NDCG@10','Recall@10','Candidate recall'],
          [[names[a],num(pooled[a]['ndcg'],6),num(pooled[a]['recall'],6),num(pooled[a]['candidate_recall'],6)] for a in names],
          [.34,.22,.22,.22], 'Равное среднее трёх недельных user-macro. Это не pooled среднее по всем query-строкам.')
        self.p('Random имеет более высокий точечный NDCG ('+num(pooled['random']['ndcg'],6)+
          '), но победитель по final не выбирается. Общий candidate recall '+
          num(pooled['manual']['candidate_recall'],6)+' показывает сильное ограничение retrieval: '
          'большинство истинных треков не попадает в union. Абсолютные метрики нельзя напрямую '
          'сопоставлять с KION, где отличаются цель, повторы и каталог.','small')
        self.fig('yambda_weekly.png',185,'Каждая неделя показана отдельно. Системы используют обновляемый прошлый префикс; наборы окон и GBDT не меняются.')
        for arm in ['importance','random']:
            c=selected[arm];windows=', '.join(g.removeprefix('window_') for g in c['groups'])
            self.p(names[arm]+': выбранные окна ['+windows+'], '+str(c['features'])+
              ' признаков, '+str(c['trees'])+' деревьев в одной seed-модели; development NDCG '+
              num(c['metrics']['ndcg@10'],6)+'.','small')
        populations=d['test_weeks']
        self.p('Число будущих активных пользователей по неделям: '+
          ', '.join(str(w['population']['query_users']) for w in populations)+
          '; без будущих positives: '+', '.join(str(w['population']['no_positive_users']) for w in populations)+
          '; cold active: '+', '.join(str(w['population']['cold_active_users']) for w in populations)+
          '. Пропущенные retrieval/catalog positives остаются ошибками системы, а не исключаются из оценки.','small')
        self.page()
    def yambda_inference(self):
        self.h('Yambda: неопределённость, устойчивость и стоимость')
        d=self.yambda
        self.p('Bootstrap ресэмплирует union пользователей с positives хотя бы в одной из трёх '
          'final-недель, а не все 1000 raw test ID. Один multiplicity пользователя применяется '
          'во всех неделях; в каждой реплике сначала вычисляется weighted macro внутри недели, '
          'затем равное среднее трёх недель. 2000 повторов, seed2026. Это сохраняет повторение '
          'пользователей и пропуски отдельных недель; никогда не активные не получают искусственную метрику.','small')
        paired=d['paired_user_bootstrap']['all_active'];rows=[]
        labels={'importance_minus_manual':'Importance - manual (основной)',
                'random_minus_manual':'Random - manual', 'importance_minus_random':'Importance - random'}
        for name in labels:
            b=paired[name];rows.append([labels[name],ci(b['ndcg']),ci(b['recall'])])
        self.table(['Контраст','ΔNDCG [95% интервал]','ΔRecall [95% интервал]'],rows,
          [.36,.32,.32], 'Clustered user-bootstrap с общими multiplicities во всех неделях; маргинальные описательные интервалы.')
        b=d['primary_result']['bootstrap']
        deltas=[w['systems']['importance']['all_active']['ndcg']-w['systems']['manual']['all_active']['ndcg'] for w in d['test_weeks']]
        positive=sum(v>0 for v in deltas);negative=sum(v<0 for v in deltas)
        stability=('Знак одинаков во всех трёх неделях. ' if positive==3 or negative==3 else
                   'Направление различается между неделями. ')
        self.p('Основной контраст importance - manual: '+ci(b)+', '+change_text(b)+'. '
          'Недельные дельты: '+', '.join(signed(v,6) for v in deltas)+
          '; положительны '+str(positive)+' из трёх, отрицательны '+str(negative)+
          '. '+stability+'Среднее и интервал относятся к этим фиксированным периодам, '
          'а не к распределению качества на всех будущих неделях.')
        self.p('В bootstrap '+str(b['users'])+' уникальных пользователей с positives хотя бы в одной '
          'из трёх недель и '+str(b['queries'])+' user-week queries. Пропуски отдельных недель '
          'сохраняются; один пользователь не превращается в независимые наблюдения.','small')
        self.fig('yambda_deltas.png',140,'Три заранее заданных контраста. Отрицательная шкала означает преимущество reference; итоговый вариант по этим данным не выбирается.')
        trials=d['trials'];cost=[]
        shared=sum(t['fit_seconds'] for t in trials if t['name'] in ['manual','all_windows'])
        for arm in ['manual','importance','random']:
            selection=sum(t['fit_seconds'] for t in trials if t['name'].startswith(arm+'_')) if arm!='manual' else 0
            models=d['final_models'][arm]
            cost.append([arm,num(selection,1),num(models['fit_seconds_all_seeds'],1),
                         num(models['model_bytes_all_seeds']/1e6,1)])
        self.table(['Вариант','Поиск окон, с','Refit 3 seeds, с','Модели, MB'],cost,
          [.27,.25,.25,.23], 'Общие development-обучения manual/всех окон: '+num(shared,1)+' с, отдельно от четырёх обучений на наборах каждой процедуры.')
        self.p('Эта внешняя проверка меняет домен, доступность полной истории и цель, а не просто '
          'пользователей KION. Она не является причинным A/B-тестом или оценкой всех будущих недель. '
          'Три периода фиксированы; bootstrap учитывает повторение пользователей, но не ресэмплирует '
          'обучение, search RNG, платформу или будущую календарную среду. Одного random seed '
          'недостаточно для вывода о распределении эффективности random search.','small')
        self.page()
    def discussion(self):
        self.h('12. Обсуждение и границы вывода')
        primary=self.test['primary'];manual=self.test['manual'];rep=self.test.get('representations')
        self.p('Исходная смесь KION даёт итоговую разность NDCG@10 '+signed(primary['ndcg@10']-manual['ndcg@10'])+
          ' относительно manual. Она объединяет две ветви, каждая сохраняет ручную базу. '
          'Её 95% интервал включает ноль; шесть моделей против трёх увеличивают измеренную стоимость '
          'запроса. Это сохранённый исходный primary-результат, а не рекомендация '
          'платить за доказанный прирост. Новый контроль и внешняя проверка не меняют её после теста.')
        if rep:
            self.p('Для конструкции representations разность относительно manual равна '+
              signed(rep['ndcg@10']-manual['ndcg@10'])+' NDCG@10. Это отдельная проверка замены '
              'части агрегатов при сохранении общих генераторов, категорий и history similarity. '
              'Её нельзя смешивать с результатом auto_selected или объявлять результатом всех '
              'возможных обучаемых представлений. Практическая приемлемость меньшего набора признаков '
              'зависит от допустимой потери качества, которая заранее не задавалась.')
            modelcost=self.data.get('costs',{}).get('models',{})
            repbytes=modelcost.get('representations',{}).get('model_bytes_all_seeds')
            manualbytes=modelcost.get('manual',{}).get('model_bytes_all_seeds')
            self.p('Ветка с 58 входами исключает явные оконные и трендовые агрегаты; сжатие истории '
              'через ALS не обязано восстанавливать ту же информацию. Это правдоподобное объяснение '
              'потери качества, а не изолированный причинный тест. Меньше признаков также не означает '
              'меньше модель: ensemble representations занимает '+num(repbytes/1e6 if repbytes is not None else None,1)+
              ' MB против '+num(manualbytes/1e6 if manualbytes is not None else None,1)+
              ' MB у manual. Глубина и число деревьев различаются. Anchors улучшают точечный результат '
              'этой компактной конструкции, но её NDCG остаётся ниже manual; прямой отдельный '
              'anchor-versus-manual интервал здесь не заявляется.')
        self.p('Anchor-конструкция имеет понятную инвариантность и ограниченную размерность, но '
          'математическая корректность не гарантирует статистической полезности. Семантика фиксированного '
          'объекта сохраняется между неделями, тогда как его факторы и окружающая геометрия могут меняться. '
          'В исходных train-факторах span якорей не охватывает всё пространство; отрицательные или '
          'нестабильные matched-дельты не следует скрывать за общим качеством наиболее сильной смеси.')
        self.sub('Ограничения валидности')
        self.p('<b>Временная переносимость.</b> Новые пользователи оцениваются на знакомой календарной '
          'неделе KION. Внешняя Yambda-проверка добавляет три заранее заданных периода другого домена '
          'без выбора по их результатам. Она расширяет проверку, но не делает эти три периода '
          'случайной выборкой всех будущих недель.')
        self.p('<b>Наблюдаемость истории.</b> Последняя дата пары скрывает ранние повторы, если её '
          'последний просмотр лежит в будущем. Статические metadata не имеют истории доступности, '
          'а ingestion delay неизвестен. Законность прошлых aggregate-полей не устраняет эти ограничения.')
        self.p('<b>Условная популяция.</b> Warm future-positive denominator отличается от всей '
          'пользовательской базы; неявные negatives и positive-containing train queries также задают '
          'конкретную выборку. Исследование не измеряет удовлетворённость, diversity, возвращаемость '
          'или причинное действие рекомендаций.')
        self.p('<b>Бюджет и множественность.</b> В исходном KION-поиске число моделей не полностью выровнено. '
          'Feature selection, HPO и смесь повторно используют две недели; bootstrap условен '
          'на выбранных системах. Дополнительный четырёх-fit random control и симметричная '
          'Yambda-процедура закрывают конкретный пробел бюджета, а не устраняют search variance. '
          'Утверждения о non-inferiority или универсальном превосходстве не следуют.')
        self.p('<b>Ресурсы.</b> Локальное wall time зависит от аппаратуры, конкурирующей нагрузки, '
          'кеша и памяти. Общий diagnostic scoring 21 модели не является latency production primary '
          'из шести моделей. Отдельный selective-serving замер сравнивает три фиксированные '
          'ветви на одном batch; заранее заданный business quality-cost критерий по-прежнему отсутствует.')
        self.page()
    def conclusions(self):
        self.h('13. Выводы и масштабирование')
        self.p(self.finding('manual','auto_selected')+' Это дополнение сильной ручной базы, '
          'ограниченное заранее заданными отношениями, агрегатами и девятью дополнительными окнами; '
          'произвольного поиска формул нет. Период KION был знаком, поэтому одного user-disjoint '
          'результата недостаточно для вывода о временной переносимости.')
        if self.random:
            self.p('При условном бюджете четырёх обучений importance даёт development NDCG '+
              num(self.random['importance']['best']['metrics']['ndcg@10'],6)+' против '+
              num(self.random['random']['best']['metrics']['ndcg@10'],6)+' у random. Это одна '
              'seed-fixed проба, выполненная после исходного теста; она не устанавливает '
              'распределение выигрыша importance и не используется для смены KION-системы.')
        if self.yambda and self.yambda.get('status')=='complete':
            b=self.yambda['primary_result']['bootstrap']
            self.p('На Yambda, с полной слушательской историей, сильной manual-базой и тремя '
              'фиксированными вариантами по три seeds, основной контраст importance - manual равен '+
              ci(b)+'. '+change_text(b).capitalize()+'. Это равное среднее трёх недель; '
              'две недельные разности отрицательны, одна положительна. Устойчивое внешнее '
              'преимущество не установлено; интервал условен на active-user union и этих периодах. '
              'Manual уже был лучше обоих обязательных непустых наборов на validation, поэтому '
              'диагностическая оценка не является рекомендацией добавлять окна вопреки этому сигналу.')
        self.p(self.finding('manual','representations')+' '+
          'Данный вариант заменяет часть агрегатов, но сохраняет обученные retrieval-сигналы, '
          'категории и history similarity. Результат не является выводом обо всех representation-моделях.')
        self.p(self.finding('representations','anchor_representations')+' '+
          self.finding('auto_full','anchor_auto_full'))
        self.p('Эти diagnostic-интервалы описательные. Геометрическая инвариантность якорей не означает '
          'универсального улучшения рекомендаций; независимо выбранное число деревьев и бюджеты поиска '
          'ограничивают причинное прочтение дельт. Пересечение интервала с нулём не устанавливает '
          'равенство или non-inferiority.')
        self.p('Dense factor retrieval для batch B и каталога I требует памяти O(BI) и вычислений '
          'O(BId) при размерности d. Для EASE память обучения O(I<sub>EASE</sub><super>2</super>), '
          'решение O(I<sub>EASE</sub><super>3</super>), здесь I<sub>EASE</sub>=5000. '
          'Материализация признаков имеет объём O(|C|F), рабочий batch - O(C<sub>batch</sub>F) '
          '= O(BKF), где B - число пользователей, K - среднее число кандидатов на пользователя. Почти миллион '
          'пользователей при каталоге около 15 тысяч объектов не доказывает масштабируемость '
          'на миллионы объектов и миллиарды событий; ANN retrieval и offline feature cache '
          'потребуют отдельного исследования компромисса качества и стоимости.')
        if self.serving:
            s=self.serving['cases']
            self.p('Измеренное время warm-запроса на одинаковых 128 пользователях: manual '+
              num(s['manual']['full_warm_request']['mean_seconds'],3)+' с, выбранные окна '+
              num(s['auto_selected']['full_warm_request']['mean_seconds'],3)+' с, исходная смесь '+
              num(s['primary']['full_warm_request']['mean_seconds'],3)+' с. Startup и process-load I/O '
              'измерены отдельно. У смеси KION интервал качества включает ноль при большей цене; '
              'её нельзя преподносить как установленную практическую пользу ансамбля.')
        self.p('Выбор окон, замена агрегатов представлениями и ансамблирование оказались разными '
          'исследовательскими вопросами. Содержательный итог включает отрицательные representation '
          'и нестабильные геометрические результаты, равный бюджет простого random-контроля '
          'и внешнюю проверку на фиксированных периодах. Они поддерживают ограниченные выводы '
          'о конкретной системе; переносимость и эксплуатационная ценность требуют собственных критериев.')
        self.page()
    def references(self):
        self.h('Литература и первичные реализации')
        refs=[
          ('Т-Банк, 2019. Разбор решения Rekko Challenge: многогенераторные рекомендации и бустинг.',
           'https://habr.com/ru/companies/tbank/articles/454818/'),
          ('Petrov A., Safilo I., Tikhonovich D., Ignatov D., 2022. MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation.',
           'https://arxiv.org/abs/2209.00325'),
          ('Тихонович Д. Открытое решение ODS MTS RecSys Challenge: ItemKNN, временные признаки и CatBoost.',
           'https://github.com/blondered/ods_MTS_RecSys_Challenge_solution'),
          ('Petrov A. KION solution: featurizer, многогенераторная схема и LambdaRank.',
           'https://github.com/asash/bert4rec_repro'),
          ('RecTools. Benchmark iALS with user/item features: KION weighting and metadata recipe.',
           'https://rectools.readthedocs.io/en/latest/examples/5_benchmark_iALS_with_features.html'),
          ('Kula M., 2015. Metadata Embeddings for User and Item Cold-start Recommendations.',
           'https://arxiv.org/abs/1507.08439'),
          ('Steck H., 2019. Embarrassingly Shallow Autoencoders for Sparse Data.',
           'https://arxiv.org/abs/1905.03375'),
          ('Qi D., Zheng W., Wang J., 2024. FeatAug: Automatic Feature Augmentation From One-to-Many Relationship Tables.',
           'https://arxiv.org/abs/2403.06367'),
          ('Featuretools. Handling Time: cutoff-aware feature calculations and training windows.',
           'https://docs.featuretools.com/en/stable/getting_started/handling_time.html'),
          ('Melchiorre A. B. et al., RecSys 2022. ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations.',
           'https://hcai.at/publications/2022_protomf/'),
          ('Ploshkin A. et al., 2025. Yambda-5B - A Large-Scale Multi-modal Dataset for Ranking And Retrieval.',
           'https://arxiv.org/abs/2505.22238'),
          ('Yandex. Official Yambda dataset card: event formats, organic flag and Listen+ definition.',
           'https://huggingface.co/datasets/yandex/yambda'),
        ]
        for index,(title,url) in enumerate(refs,1):
            self.p(f'[{index}] '+escape(title)+f'<br/><link href="{escape(url)}" color="#203e51">'+escape(url)+'</link>','small')
        self.sub('Материалы исследования')
        self.p('Числа таблиц и рисунков формируются из results/experiment.json, random_search.json, '
          'serving_benchmark.json и yambda.json. Это сохранённые агрегированные метрики, выбранные '
          'модели, парные интервалы, ресурсы и диагностика кандидатов/якорей. Статья и графики '
          'не выполняют обучение, подбор моделей '
          'или новое вычисление рекомендаций. Дельты и суммы ресурсов являются арифметикой уже '
          'сохранённых агрегатов. Индивидуальные истории и списки пользователей для чтения статьи не требуются.')
        hardware=self.data.get('hardware',{})
        if hardware:self.p('Исходные локальные измерения выполнены на macOS (Darwin), архитектура arm64: '
          'Python 3.11.15, 10 логических CPU, 16 GiB оперативной памяти. Для GBDT использованы '
          '4 потока, для collaborative-моделей и BLAS - 1 поток. Это описание исследовательского '
          'окружения, а не оценка throughput выделенного production-сервера.','small')
        self.p('Датировка исходной выгрузки соответствует использованным CSV 2021 года, а не '
          'календарной опечатке в README набора. Наблюдаемые свойства benchmark и преобразования '
          'в этой работе отделены от результатов исходных авторов. Опубликованные бизнес-эффекты '
          'и метрики конкурсных решений не переносятся на полученные здесь оценки.','small')
    def build(self):
        self.title();self.related();self.dataset();self.retrieval();self.windows();self.representations()
        self.learning();self.development();self.random_control();self.final_results();self.uncertainty();self.costs()
        self.serving_results();self.yambda_protocol();self.yambda_methods();self.yambda_results();self.yambda_inference()
        self.discussion();self.conclusions();self.references()
        self.output.parent.mkdir(parents=True,exist_ok=True)
        def footer(canvas,doc):
            canvas.saveState();canvas.setFont('Article',8)
            if doc.page>1:canvas.drawString(22*mm,A4[1]-15*mm,'Временные признаки и представления: KION и Yambda')
            canvas.drawCentredString(A4[0]/2,14*mm,str(doc.page));canvas.restoreState()
        doc=SimpleDocTemplate(str(self.output),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,
          topMargin=22*mm,bottomMargin=22*mm,title=TITLE,author='',subject='KION и Yambda: временная оценка, равный бюджет и качество-стоимость')
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        return self.output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/experiment.json'))
    parser.add_argument('--output',type=Path,default=Path('article.pdf'))
    parser.add_argument('--random',type=Path)
    parser.add_argument('--serving',type=Path)
    parser.add_argument('--yambda',type=Path)
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args();print(Article(args.input,args.output,args.random,args.serving,args.yambda,args.figures).build())
