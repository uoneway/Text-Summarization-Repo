# Text Summarization Repo
NLP 중에서도 텍스트 요약 관련 다양한 자료를 축적해나가는 공간입니다. 텍스트 요약 분야에 관심 있어 공부를 시작하시는 분들께 좋은 길잡이가 되면 좋겠습니다.

우선은 텍스트 요약 분야의 분류, 주요 연구주제 등 이 분야를 공부하기 전에 알아둬야 할 필수 지식들을 정리했습니다. 그리고 이 분야 흐름을 이해하기 위해 필수적으로 읽어야 할 논문들과, 한국어/영어 요약 모델을 구성하기 위해 이용할 수 있는 Data와 Pre-trained Model, 이들을 이해하기 위해 참고할만한 양질의 자료들을 목록화 했습니다.

  * [Intro to Text Summarization](#intro-to-text-summarization)
      * [Definition](#definition)
      * [Task Categories](#task-categories)
      * [Main Topics](#main-topics)
      * [Prerequisite](#prerequisite)
  * [Resources](#resources)
    + [Must-read Papers](#must-read-papers)
    + [SOTA & Latest Research List](#sota--latest-research-list)
  * [Data & Pre-trained Models](#data--pre-trained-models)
    + [Korean](#korean)
      - [Data & Competitions](#data--competitions)
      - [Pre-trained Models](#pre-trained-models)
    + [English](#english)
      - [Data & Competitions](#data--competitions-1)
  * [Others](#others)
    + [Services](#services)
    + [Resources](#resources-1)
    + [Recommended Papers list](#recommended-papers-list)



## Intro to Text Summarization

### Definition

 [Berry, Dumais, & O’Brien (1995)](https://epubs.siam.org/doi/abs/10.1137/1037127)은 텍스트 요약을 다음과 같이 정의합니다.

> Text summarization is the process of **distilling the most important information** from a text to produce an abridged version for a particular task and user

한 마디로 **주어진 텍스트 중 중요한 정보만 정제해내는 과정**이라는 건데요. 여기서 *정제*라는 표현과 *중요한*이라는 표현은 다소 추상적이고 주관적인 표현이기에 개인적으로는 다음과 같이 정의하고 싶습니다.

**`f(text) = comprehensible information`**

즉 텍스트 요약은 **원문을 이해하기 쉬우면서도 가치있는 정보로 변환**하는 작업입니다. 인간은 길이가 길거나 여러 문서로 나눠져있는 텍스트 정보를 한 눈에 파악하기 어려워합니다. 때로는 알지 못하는 전문 용어가 많이 사용되어 있을 수도 있습니다. 이러한 텍스트를 원문을 잘 반영하면서도 간결하여 이해하기 쉬운 형태로 바꿔주는 작업은 상당히 가치있는 일입니다. 물론 무엇이 정말 가치있는 정보인지, 어떠한 형태로 바꿔줄지 등은 요약을 하는 목적이나 개인의 취향에 따라 달라지게 될 것입니다. 

이러한 관점으로 바라본다면 회의록, 신문기사 헤드라인, 논문 abstract, 이력서 등 텍스트를 생성해내는 task뿐만 아니라 텍스트를 그래프나 영상으로 변환하는 작업들도 텍스트 요약에 해당된다고 말할 수 있습니다. 물론 그냥 summarization이 아닌 *Text Summarization*이기에 요약의 대상(source)은 text형태로 한정됩니다. (요약의 source는 text뿐만 아니라 image나 video가 될 수도 있기 때문이죠. 전자의 예가 image captioning이고, 후자의 예가 [Video Summarization](https://paperswithcode.com/task/video-summarization)입니다. 물론 Vision과 NLP의 경계가 점점 흐릿해져가는 최근 딥러닝 추세를 고려해본다면, 곧 'Text'를 pefix로 붙이는게 의미 없는일이 되지는 않을까 하는 생각이 들기도 하네요.)

### Task Categories

일반적으로 텍스트 요약  task는 크게 요약문을 생성하는 방식에 따라 **extractive summarization**(이하 ext)과 **abstractive summarization**(이하 abs)로 나눕니다. ([Gudivada, 2018](https://www.sciencedirect.com/topics/computer-science/text-summarization))

> Extractive methods select a subset of existing words, phrases, or sentences in the original text to form a summary. In contrast, abstractive methods first build an internal semantic representation and then use natural language generation techniques to create a summary. 

**Ext**는 보통 **문장 단위로 중요도를 scoring한 후, 이를 기반으로 선택하고 조합**하여 summary를 만듭니다. 우리가 글을 읽다가 형광펜을 칠하는 작업과 비슷하죠. 반면 **abs**는 **원문을 기반으로 하되, 새로운 텍스트(novel text)를 생성해내는 NLG(natural language generation) 방식**입니다. Ext는 원문에 있던 텍스트를 활용하기에 표현이 제한적이나 말이 안되는 표현이 포함될 가능성이 낮습니다. 반면 abs는 모델에서 새로운 텍스트를 생성해내야 하기에 말이 되지 않는 표현이 만들어질 가능성이 존재하나 좀더 flexible한 접근이 가능하다는 장점이 있습니다.

이 외에도 원문의 개수에 따라 **single/multi document** summarization, 생성해내는 텍스트 형태에 따라 **keyword/sentence** summarization, 요약 과정에서 원문 외 외부 정보를 얼마나 사용하는지에 따라 **knowlege-poor/rich** summarization 등 다양한 구분이 있습니다.

![Figure 2.1: Classification of summarization tasks.](images/Classification_of_summarization_tasks.png)

(G. Sizov(2010). [Extraction-Based Automatic Summarization: Theoretical and Empirical Investigation of Summarization Techniques](https://www.semanticscholar.org/paper/Extraction-Based-Automatic-Summarization%3A-and-of-Sizov/2d27fd9af4b10cc5b54a849a3c2ad84755b3b13c))



### Main Topics

Text Summarization 분야의 주요 연구주제를 살펴보고 이 분야에 어떤 Challenge가 존재하는지 함께 생각해봅시다.

- **Multi / Long documents summarization**

  앞서 언급했듯 요약이라는 task는 incomprehensible text를 comprehensible information로 바꾸는 작업입니다. 그렇기에 원문이 길어질수록, 또는 한 번에 한 문서가 아닌 여러 소스의 문서를 요약할수록 요약의 효용은 증가합니다. 문제는 동시에 요약 난이도 또한 증가한다는 점이겠죠.

  그 이유로는 첫째, 긴 원문을 처리하려면  computational complexity가 증가합니다. 이는 과거의 TextRank 같은 통계 방식에서 보다, 최근 transformer를 위시한 신경망 기반 방식에서 훨씬 더 critical한 문제입니다. 둘째, 원문이 길수록 그 안에 핵심이 아닌 내용, 즉 noise가 많이 포함되어 있기 마련입니다. 무엇이 noise고 무엇이 informative한 텍스트인지 가려내기가 쉽지 않습니다. 마지막으로 긴 원문이나 다양한 소스는 다양한 관점과 내용을 동시에 가지고 있기에 이를 잘 포괄하는 요약문을 생성하는 것이 어려워지죠. 

  - **Multi documents summarization(MDS)**

    MDS는 **복수개의 문서를 요약**하는 작업입니다.  [Liu et al. (2018)](https://arxiv.org/abs/1801.10198)처럼 위키 문서를 생성해내는 작업을 MDS로 생각해볼 수 있습니다. 이 때 위키에 reference로 달린 웹사이트 본문들이 원문, 생성되는 위키문서가 summary가 됩니다. 요약하는 대상이 리뷰 모음이라면 이 또한 텍스트 길이가 짧고 주관성이 높다는 특징을 가진 MDS의 일종이(이를 Opinion summarization라고도 부릅니다).

  - **Long documents summarization** 

    [Liu et al. (2018)](https://arxiv.org/abs/1801.10198)는 긴 텍스트를 인풋으로 받아들이기 위해 우선 통계적 방법으로 extractive summary를 만들어 중요한 문장만 추린 후 모델의 입력으로 사용합니다. 또한 transformer 연산량을 줄이기 위해 input을 블락 단위로 나눠서 연산하고 이 때 1-d convolution을 적용하여 개별 attention key, value의 수를 줄인 attention 방식을 사용합니다. [Big Bird (2020)](https://arxiv.org/abs/2007.14062) 논문은 transformer의 계산량을 줄이기 위해 기존 모든 단어 간 조합을 살펴보는 full attention 방식(quadratic) 대신 sparse attention mechanism(linear)을 도입합니다. 그 결과 동일 성능 하드웨어로 최대 8배까지 긴 문장을 요약할 수 있었습니다. 

    반면 [Gidiotis &Tsoumakas (2020)](https://arxiv.org/abs/2004.06190)은 긴 텍스트 요약 문제를 한 번에 풀지 않고 여러 작은 텍스트 요약 문제들로 바꿔 푸는 divide and-conquer 접근을 시도합니다. 원문과 target summary를 multiple smaller source-target pairs로 바꿔서 모델을 훈련합니다. inference 시에는 이 모델을 통해 출력된 partial summaries를 aggregate하여 complete summary를 만듭니다. 

- **Performance improvement**

  어떻게 하면 더 좋은 요약문을 생성해낼 수 있을까요?

  - **Transfer Learning**

    최근 NLP에서 Pretraining model을 이용하는 것은 거의 default가 되었습니다. 그렇다면 Text Summarization에서 좀 더 좋은 성능을 보여줄 수 있는 Pretraining model을 만들기 위해서는 어떤 구조를 가져야 할까요? 어떤 objective를 가져야 할까요?

    [PEGASUS (2020)](https://arxiv.org/abs/1912.08777)에서는 텍스트 요약과정과 objective가 유사할수록 높은 성능을 보여줄 것이라는 가정하에  ROUGE score에 기반하여 중요하다고 판단되는 문장을 골라 문장 단위로 마스킹하는 GSG(Gap Sentences Generation) 방식을 사용했습니다. 현 SOTA 모델인 [BART (2020)](https://arxiv.org/abs/1910.13461)(Bidirectional and Auto-Regressive Transformers)는 입력 텍스트 일부에 노이즈를 추가하여 이를 다시 원문으로 복구하는 autoencoder 형태로 학습합니다.

  - **Post-editing Correction**

    한 번에 좋은 요약문을 생성해내면 좋겠지만, 쉽지 않은 일입니다. 그렇다면 우선 요약문을 생성한 후 이를 다양한 기준에서 검토하고 수정해보면 어떨까요?

    일례로 [Cao, Dong, Wu, &Cheung (2020)](https://arxiv.org/abs/2010.08712)은 생성된 요약문에 pretrained neural corrector model을 적용하여 Factual Error를 감소시키는 방법을 제시합니다.

  - 이 외에도 최근 핫한 Graph Neural Network(GNN)을 적용한 시도들도 많아지고 있습니다.

- **Metric** / **Evaluation method**

  앞서 '좋은'이라는 두루뭉술한 표현을 썼는데요. 과연 '좋은 요약문'이란 무엇일까요?  [Brazinskas, Lapata, & Titov (2020)](https://arxiv.org/abs/2004.14884)에서는 좋은 요약문의 판단기준으로 다음 다섯가지를 사용합니다.

  - **Fluency**:  the summary should be grammatically correct, easy to read and understand; 
  - **Coherence**: the summary should be well structured and well organized; 
  - **Nonredundancy**: there should be no unnecessary repetition in the summary; 
  - **Informativeness**: how much useful information about the product does the summary provide?
  - **Sentiment**: how well the sentiment of the summary agrees with the overall sentiment of the original reviews?

  문제는 이러한 부분을 측정하는 것이 쉽지 않다는 점입니다. 텍스트 요약에서 가장 흔히 사용되는 성능 측정 지표는 ROUGE score인데요. ROUGE score에는 다양한 변종이 있지만, 기본적으로 'generated summary와 reference summary의 출현 단어와 그 순서가 얼마나 일치하는가'를 측정합니다. 뜻은 유사하지만 형태는 다른 단어가 나오거나, 단어 순서가 달라지면 설혹 더 좋은 요약문일지라도 낮은 점수를 받을 수 있겠죠. 특히 ROUGE score를 높이려다 오히려 요약문의 표현적 다양성(diversity)을 헤치는 결과를 가져올 수 있습니다. 이 때문인지 많은 논문에서는 ROUGE score 뿐만 아니라 비싼 돈이 들어가는 human evaluation 결과를 추가로 제시합니다.  

  [Lee et al. (2020)](https://arxiv.org/abs/2005.03510)은 generated summary가 본문 및 reference summary와 얼마나 유사한지를 이들을 SBERT로 임베딩한 후 나오는 벡터 간 유사도로 측정하는 RDASS(Reference and Document Awareness Semantic Score)를 제시합니다. 이러한 방식은 특히 단어와 여러 형태소가 결합하여 다양한 의미와 문법적 기능을 표현하는 교착어인 한국어 평가 정확도를 올려줄 것으로 기대됩니다. [Kryściński, McCann, Xiong, & Socher (2020)](https://arxiv.org/abs/1910.12840)는 Factual Consistency를 평가하기 위한 weakly-supervised, model-based approach를 제안했습니다.

- **Controllable text generation**

  주어진 문서에 대해 최고의 요약문은 오직 하나일까요? 그렇지 않을 겁니다. 성향이 다른 사람들은 동일한 원문에 대해서도 서로 다른 요약문을 더 선호할 수 있습니다. 혹 동일한 사람일지라도  요약을 하는 목적이나 상황에 따라 원하는 요약문은 달라지겠죠. 이렇게 **사용자가 지정한 조건에 따라 출력을 원하는 형태로 조정하는 방식**을 *controllable text generation*라고 합니다. 주어진 문서에 대해 동일한 요약문을 생성하는 *Generic summarization*에 비해 **개인화**된 요약문을 제공할 수 있습니다. 

  이렇게 생성되는 요약문은 이해하기 쉽고 가치있어야 할 뿐만 아니라 **함께 넣어준 condition과 밀접하게 연관성을 지녀야 합니다.**

  `f(text, `**`condition`**`) = comprehensible information`**`that meets the given conditions`**

  요약모델에 어떤 *condition*을 추가해볼 수 있을까요? 그리고 어떻게 하면 그 조건에 맞는 요약문을 생성해낼 수 있을까요?

  - **Aspect-based summarization**

    에어팟 사용자 리뷰를 요약할 때 음질, 배터리, 디자인 등의 측면을 나눠 각각에 대한 요약을 하고 싶을 수 있습니다. 아니면 글의 writing style이나 sentiment를 조정해보고 싶을 수도 있습니다. 이렇게 원문에서도 **특정한 측면(aspect) 또는 특성(feature)과 관련있는 정보만을 요약하는 작업**을 *aspect-based summarization*라고 합니다.

    기존에는 주로 모델 학습 시 사용했던 pre-defined aspect에서만 제한적으로 작동하는 모델만 가능했지만 최근들어 [Tan, Qin, Xing, & Hu(2020)](https://arxiv.org/abs/2010.06792)와 같이 학습 시 주어지지 않았던 **arbitrary aspect에 대한 추론**을 가능케 하기 위한 시도들이 이루어지고 있습니다.  

  - **Query focused summarization**(QFS)

    Condition*이 *query*인 경우, QFS라 부릅니다. **Query는 주로 자연어 형태**를 띠고 있기에 이 다양한 표현들을 어떻게 잘 임베딩하여 원문과 매칭할지가 주요한 과제입니다. 우리가 잘 아는 QA시스템과 상당히 유사하다고 할 수 있습니다. 
    
  - **Update summarization**

    인간은 계속해서 학습하고 성장해나가는 동물입니다. 따라서 특정 정보에 대해 내가 느끼는 오늘의 가치는 일주일 후의 가치와 전혀 다를 수 있습니다. 내가 이미 경험한 문서에 있던 내용의 가치는 낮아질 것이고, 아직 경험하지 못한 새로운 내용은 여전히 높은 가치를 지닐 것입니다. 이러한 관점에서 **해당 이용자가 이전에 경험했던 문서 내용과 유사도가 낮은 새로운 내용 위주의 요약문을 생성하는 작업**을 *update summarization*이라고 합니다.

- **Knowledge-enhanced text generation**

  Text-to-text task에서는 원문만으로는 원하는 출력을 생성하는 것이 어려운 경우가 많습니다. 그래서 **원문뿐만 아니라 추가로 다양한 knowledge를 모델에 제공하여 성능을 높이려는 시도**가 있습니다. 이러한 knowledge의 source나 제공 형태는 keywords, topics, linguistic features, knowledge bases, knowledge graphs, grounded text 등 다양합니다.

  예를 들어 [Tan, Qin, Xing, & Hu(2020)](https://arxiv.org/abs/2010.06792)는 일반적인 summry dataset을 복수개의 Aspect-based Summary로 변환하기 위해 knowledge graph인 ConcepNet을, 주어진 aspect와 관련한 더 풍성한 정보를 모델에 전달하기 위해 Wikipedia를 활용합니다. 보다 더 자세히 알고 싶다면 [Yu et al. (2020)](https://arxiv.org/abs/2010.04389)가 쓴 survey 논문을 읽어보세요.

- **Data scarcity problem**

  텍스트 요약이란 task는 사람이 하기에도 쉽지 않은, 시간이 많이 소모되는 작업입니다. 따라서 dataset을 만드는데 상당히 큰 비용이 소모되고 당연히 training을 위한 데이터가 많이 부족합니다.

  그래서 앞서 언급한 Pretraining model을 이용하는 방식 외에도 **Few-Shot Learning**적 접근을 시도하거나 **Reinforcement learning**방식을 적용하려는 시도가 있습니다. 또한 좋은 요약 데이터를 만드는 것 자체도 굉장히 중요한 연구 주제입니다.

- 이 외에도 **모델 경량화**와 같은 전형적인 DL 주제는 물론이고 뉴스나 위키백과와 같은 Structured text가 아닌 대화체(Conversational Structure) 등에 적합한 요약모델을 만들고자 하는 시도 등 다양한 주제들이 있습니다.  

### Prerequisite

텍스트 요약 분야 공부를 시작하는데 있어 다음을 알고 있다면 좀 더 쉽게 공부를 진행해나갈 수 있을 것입니다.

- NLP 기본 개념 이해

  - Embedding
  - Transfer learning(Pre-training  + Fine-tunning)

- Transformer/BERT 구조 및 Pre-training objective 이해

  최신 NLP 분야 논문들의 상당수가 Transformer에 기반하여 만들어진 BERT, 그리고 이 BERT의 변형인 RoBERTa, T5 등 여러 Pretraining model에 기반하고 있습니다. 따라서 이들의 개략적 구조와  Pre-training objective에 대해 얕게나마 이해하고 있다면 논문을 읽거나 구현함에 있어 큰 도움이 됩니다. 

  - [글] [구상준(PINGPONG 블로그). Transformer - Harder, Better, Faster, Stronger: Transformer](https://blog.pingpong.us/transformer-review/)
  - [영상] [이유경(KoreaUniv DSBA) . Transformer to T5 (XLNet, RoBERTa, MASS, BART, MT-DNN,T5)](https://www.youtube.com/watch?v=v7diENO2mEA)

- Graph Neural Network(GNN)

  - [영상] [강현규(KoreaUniv DMQA). Graph Attention Networks](http://dmqm.korea.ac.kr/activity/seminar/296)

- Text Summarization 기초 개념

  - Summarization 기본 용어
    - *Original text* = *Source text*
    - *generated summary*는 모델이 생성해낸 요약문을 의미합니다. 반면 우리가 정답으로 간주하는(보통은 사람이 직접 원문을 보고 생성한) 요약문은 *reference summary* 또는 *gold summary*라고 부릅니다. 보통은 두 용어를 크게 구분없이 쓰는듯 하나, 전자는 *generated summary*를 평가하기 위한 기준이 되는 요약문이라는 면을 강조할 때, 후자는 우리가 찾는 진짜 요약문이라는 점을 강조할 때 주로 사용되는 듯 합니다.  
  - Metric: Rouge, BLEU, Perplexity(PPL) 등
  - [글] [icoxfog417/awesome-text-summarization](https://github.com/icoxfog417/awesome-text-summarization)
  - [PPT] [Sang-Houn Choi. Text summarization](https://www.slideshare.net/cozyhous?utm_campaign=profiletracking&utm_medium=sssite&utm_source=ssslideview)



## Resources

### Must-read Papers

| Year | Paper                                                        | Keywords                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2004 | [**TextRank**: Bringing order into texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)<br />R. Mihalcea, P. Tarau<br />*추출요약 분야의 고전이자 현재도 활발히 사용되고 있는 대표적인 모델로,  통계(PageRank) 기반unsupervised learning 방식으로 별도의 학습 데이터 없이 추론이 가능하고, 알고리즘이 명확하여 이해가 쉽습니다* <br /><br />- [Library] [gensim.summarization](https://radimrehurek.com/gensim_3.8.3/auto_examples/tutorials/run_summarization.html#sphx-glr-auto-examples-tutorials-run-summarization-py)(3.x버전만 가능. 4.x버전에서 삭제),  [PyTextRank](https://github.com/DerwenAI/pytextrank)<br />- [이론/Code] [lovit. TextRank 를 이용한 키워드 추출과 핵심 문장 추출](https://lovit.github.io/nlp/2019/04/30/textrank/) | Model(ext),<br />Graph-based(PageRank),<br />Unsupervised    |
| 2019 | **BertSum**: [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) ([Code](https://github.com/nlpyang/PreSumm))<br/>Yang Liu, Mirella Lapata / EMNLP 2019<br />*Pre-trained BERT를 요약 task에 활용하려면 어떻게 해야할까요? BertSum은 이를 위해  BERT에 Transformer layers를 얹은 구조를 사용합니다. 또한 여러 sentence를 하나의 인풋으로 넣어주기 위해 매 문장 앞에 [CLS] 토큰을 삽입하고 interval segment embeddings을 추가한 변형 input embeddings을 제안합니다. ext모델의 output을 abs모델의 input으로 이용하는 2staged fine-tuning 접근도 보여줍니다.*<br /><br />- [Review] [이정훈(KoreaUniv DSBA)](https://www.youtube.com/watch?v=PQk9kr9dGu0)<br />- [Code] [KoBertSum(수정중)](https://github.com/uoneway/KoBertSum) | Model(ext/abs), <br />BERT+transformer,<br />2staged fine-tuning |
| 2019 | [**BART**: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://www.aclweb.org/anthology/2020.acl-main.703/) <br />Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer / ACL 2020<br />*BERT는 bidirectional encoder로 generation task에 약하고, GPT는 auto-regressive한 model로 bidirectional한 정보를 얻지 못한다는 단점을 가집니다. BART는 이들을 결합한 seq2seq 형태를 가짐으로 기존에 나왔던 다양한 nosing기법을 한 모델에서 실험해볼 수 있었습니다. 그 결과 Text infilling(text span을 하나의 mask token으로 바꿈)과 Sentence shuffling(문장 순서를 랜덤으로 섞음)을 동시에 적용함으로, 특히 summarization 분야에서 기 SOTA 모델을 훌쩍 뛰어넘는 성능을 보여줍니다.*  <br /> <br />- [Code] SKT T3K. **[KoBART](https://github.com/SKT-AI/KoBART)**<br />- [Review] [임연수](https://dladustn95.github.io/nlp/BART_paper_review/),  [Jiwung Hyun](https://medium.com/@kabbi159/acl-2020-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-7a0ae37109dc), | Pretraining Model(abs),<br />seq2seq,<br />Denoising autoencoder,<br />Text infilling |
| 2020 | [**MatchSum**: Extractive Summarization as Text Matching](https://arxiv.org/abs/2004.08795) ([Code](https://github.com/maszhongming/MatchSum))<br />Ming Zhong, Pengfei Liu, Yiran Chen, Danqing Wang, Xipeng Qiu, Xuanjing Huang / ACL 2020<br />- [Review] [이유경(KoreaUniv DSBA)](https://www.youtube.com/watch?v=8E2Ia4Viu94&t=1582s) | Model(ext)                                                   |
| 2020 | [Summarizing Text on Any Aspects: A Knowledge-Informed Weakly-Supervised Approach](https://arxiv.org/abs/2010.06792) ([Code](https://github.com/tanyuqian/aspect-based-summarization))<br />Bowen Tan, Lianhui Qin, Eric P. Xing, Zhiting Hu / EMNLP 2020<br />*Aspect-based summarization는 1) multiple aspect-based summaries data가 부족하고, 2) 모델을 학습한다 해도 학습한 data의 pre-defined aspects에서만 제한적으로 작동한다는 점에서 쉽지 않은 task입니다. 본 논문은 이를 external knowledge sources를 활용하여 보완합니다. 일반적인 summry dataset을 복수개의 Aspect-based Summary로 변환하기 위해 ConcepNet을, 주어진 aspect와 관련한 더 풍성한 정보를 모델에 전달하기 위해 Wikipedia를 활용합니다. 이러한 방식으로 BART를 fine-tuning함으로, 적은 데이터로 arbitrary aspect에 대해서도  우수한 성능을 보였습니다.* | Aspect-based,<br />Knowlege-rich                             |

### SOTA & Latest Research List

#### SOTA Model List

- [Papers with Code: Best method for each benchmarks](https://paperswithcode.com/task/text-summarization)
- [NLP-progress](https://github.com/sebastianruder/NLP-progress)

#### Latest Research List

- [Paper Digest: Recent Papers on Text Summarization](https://www.paperdigest.org/2020/08/recent-papers-on-text-summarization/)

- [Papers with Code: Latest papers](https://paperswithcode.com/task/text-summarization/latest#code)

- [EMNLP 2020 Papers-Summarization](https://github.com/roomylee/nlp-papers-with-arxiv/tree/master/emnlp-2020#summarization)



## Data & Pre-trained Models

아래 사용한 약자의 의미는 다음과 같습니다.

* `w`: The number of words의 평균값;  `s`: The average number of sentences의 평균값

  예) `13s/214w → 1s/26w` 는 평균 13문장(평균 214단어)으로 구성된 본문 텍스트와 평균 1개 문장(평균 26개 단어)로 구성된 요약 텍스트가 제공된다는 의미입니다.

* `abs`: Abstractive summary; `ext`: Extractive summary

### Korean

#### Data & Competitions

| Dataset                                                      | Domain / Length                                    | Volume<br />(pair)                | License                                                      |
| ------------------------------------------------------------ | -------------------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| [**모두의 말뭉치-문서 요약 말뭉치**](https://corpus.korean.go.kr/) ([명세](https://rlkujwkk7.toastcdn.net/NIKL_SUMMARIZATION(v1.0).pdf))<br />*짧은 뉴스 본문에 대한 제목, 3문장 abs 및 ext summay 제공. id로 신문 말뭉치([명세](https://rlkujwkk7.toastcdn.net/NIKL_NEWSPAPER(v1.0).pdf))와 결합하여 소제목, media, date, topic 관련 추가 정보를 얻을 수 있음* | 뉴스<br />\- origin → 3s(abs); 3s(ext)             | 13,167                            | 국립국어원<br />(개별 약정)                                  |
| [**sae4K**](https://github.com/warnikchow/sae4k)             |                                                    | 50,000                            | [CC-BY-SA-4.0](https://github.com/warnikchow/sae4k/blob/master/LICENSE) |
| **[sci-news-sum-kr-50](https://github.com/theeluwin/sci-news-sum-kr-50)** | 뉴스(IT/과학)<br />                                | 50                                | MIT                                                          |
| **[Bflysoft 구축-뉴스기사 데이터셋 (2020)](https://dacon.io/competitions/official/235671/data/)**<br />*짧은 뉴스 본문에 대한 media, 단문 abs 및 ext summay 제공*<br /><br />- [EDA] [데이터 EDA 노트북](https://github.com/uoneway/KoBertSum/blob/master/tutorials/EDA.ipynb)<br />- [한국어 문서 추출요약 AI 경진대회(~ 2020.12.09)](https://dacon.io/competitions/official/235671/overview/)<br />- [한국어 문서 생성요약 AI 경진대회(~ 2020.12.09)](https://dacon.io/competitions/official/235673/overview/) | - 뉴스<br />\- 13s/214w → 1s/26w(abs); 3s/55w(ext) | 43,000                            | 개별 약정<br />* 추후 [AIHub](https://www.aihub.or.kr/aidata/8054)를 통해 공개예정 |
| [**WikiLingua**: A Multilingual Abstractive Summarization Dataset (2020)](https://github.com/esdurmus/Wikilingua) <br />*[wikiHow](https://www.wikihow.com/)를 통해 구축한 Korean, English 등 17개국어에 대한 abs summay*<br /><br />- [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.360/), [Collab notebook](https://colab.research.google.com/drive/1HxonmcM7EOQVal2I6oTi9QWEP257BgDP?usp=sharing) | - How-to docs<br />- 391w→ 39w                     | 12,189<br />(전체 770,087 중 kor) | 2020,<br />[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) |



#### Pre-trained Models

| Model                                                        | Pre-training                                                 | Usage                                                        | License                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| [**BERT(multilingual)**](https://github.com/google-research/bert/blob/master/multilingual.md)<br />BERT-Base(110M parameters) | - Wikipedia(multilingual)<br />- WordPiece. <br />- 110k shared vocabs | - [`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) 버전 권장<br />(`--do_lower_case=false` 옵션 넣어주기)<br />- Tensorflow | Google<br />(Apache 2.0)                     |
| [**KOBERT**](https://github.com/SKTBrain/KoBERT)<br />BERT-Base(92M parameters) | - 위키백과(문장 5M개), 뉴스(문장 20M개)<br />- [SentencePiece](https://github.com/google/sentencepiece)<br />- 8,002 vocabs(unused token 없음) | - PyTorch<br />- [KoBERT-Transformers(monologg)](https://github.com/monologg/KoBERT-Transformers)를 통해 Huggingface Transformers 라이브러리로 이용 가능, [DistilKoBERT](https://github.com/monologg/DistilKoBERT) 이용 가능 | SKTBrain<br />(Apache-2.0)                   |
| [**KorBERT**](https://aiopen.etri.re.kr/service_dataset.php)<br />BERT-Base | - 뉴스(10년 치), 위키백과 등 23GB<br />- [ETRI 형태소분석 API](https://aiopen.etri.re.kr/service_api.php) / WordPiece(두 버전을 별도로 제공)<br />- 30,349 vocabs<br />- Latin alphabets: Cased<br />- [소개] [임준(ETRI). NLU Tech Talk with KorBERT](https://www2.slideshare.net/LGCNSairesearch/nlu-tech-talk-with-korbert) | - PyTorch, Tensorflow <br/>                                  | ETRI<br />(개별 약정)                        |
| **[KcBERT](https://github.com/Beomi/KcBERT)**<br />BERT-Base/Large | - 네이버 뉴스 댓글(12.5GB, 8.9천만개 문장)<br />(19.01.01 ~ 20.06.15 기사 중 댓글 많은 기사 내 댓글과 대댓글)<br />- [tokenizers](https://github.com/huggingface/tokenizers)의 BertWordPieceTokenizer<br />- 30,000 vocabs |                                                              | [Beomi](https://github.com/Beomi)<br />(MIT) |
| **[KoBART](https://github.com/SKT-AI/KoBART)**<br />[BART](https://arxiv.org/pdf/1910.13461.pdf)(124M) | - 위키백과(5M), 기타(뉴스, 책, 모두의 말뭉치 (대화, 뉴스, ...), 청와대 국민청원 등 0.27B)<br />- [tokenizers](https://github.com/huggingface/tokenizers)의 Character BPE tokenizer<br />- 30,000 vocabs(<unused> 포함)<br /><br />- [Example] seujung. KoBART-summarization([Code](https://github.com/seujung/KoBART-summarization), [Demo](http://20.194.43.11:7874/)) | - 요약 task에 특화<br />- Huggingface Transformers 라이브러리 지원<br />- PyTorch | SKT *T3K*<br />(modified MIT)                |

- 기타
  - https://github.com/snunlp/KR-BERT
  - https://github.com/tbai2019/HanBert-54k-N



### English

#### Data & Competitions

[기타 요약 관련 영어 데이터셋 명칭, domain, task, paper 등](http://pfliu.com/pl-summarization/summ_data.html)

| Dataset                                                      | Domain / Length                                              | Volume                 | License                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| **[ScisummNet](https://cs.stanford.edu/~myasu/projects/scisumm_net/)**([paper](https://arxiv.org/abs/1909.01716))<br />*ACL(computational linguistics, NLP) research papers에 대한 세 가지 유형의 summary(논문 abstract, collection of citation sentences, human summary) 제공* <br /><br />- CL-SciSumm 2019-Task2([repo](https://github.com/WING-NUS/scisumm-corpus), [paper](https://arxiv.org/abs/1907.09854))<br />- [CL-SciSumm @ EMNLP 2020-Task2](https://ornlcda.github.io/SDProc/sharedtasks.html#clscisumm)([repo](https://github.com/WING-NUS/scisumm-corpus)) | - Research paper<br />(computational linguistics, NLP)<br />- 4,417w → 110w(논문abstract) ; 2s(citation); 151w(abs) | 1,000(abs/ ext)        | [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) |
| **[LongSumm](https://github.com/guyfe/LongSumm)**<br />*NLP 및 ML 분야 Research paper에 대한 상대적으로 장문의 summary(관련 blog posts 기반 abs, 관련 conferences videos talks 기반 ext) 제공*<br /><br />- [LongSumm 2020@EMNLP 2020](https://ornlcda.github.io/SDProc/sharedtasks.html#longsumm)<br />- [LongSumm 2021@ NAACL 2021](https://sdproc.org/2021/sharedtasks.html#longsumm) | - Research paper(NLP, ML)<br />- origin → 100s/1,500w(abs); 30s/ 990w(ext) | 700(abs) +  1,705(ext) | [Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| **[CL-LaySumm](https://github.com/WING-NUS/scisumm-corpus/blob/master/README_Laysumm.md)**<br />*NLP 및 ML 분야 Research paper에 대해 비전문가를 위한 쉬운(lay) summary 제공*<br /><br />- [CL-LaySumm @ EMNLP 2020](https://ornlcda.github.io/SDProc/sharedtasks.html#laysumm) | - Research paper(epilepsy, archeology, materials engineering)<br />- origin → 70~100w | 600(abs)               | 개별약정 필요([a.dewaard@elsevier.com](mailto:a.dewaard@elsevier.com) 로 이메일을 송부) |
| [**Global Voices**: Crossing Borders in Automatic News Summarization (2019)](http://opus.nlpl.eu/GlobalVoices.php)<br />- [paper](https://www.aclweb.org/anthology/D19-5411.pdf) | - 뉴스<br />- 359w→ 51w                                      |                        |                                                              |



## Others

### Services

- [Semantic Scholar](https://tldr.semanticscholar.org/): 논문에 대한 요약서비스 제공
- [세줄요약기](https://summariz3.herokuapp.com/): TextRank 기반 세줄 요약기

### Resources

- [KoreaUniv DSBA](https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg/playlists)

- [neulab/Text-Summarization-Papers](https://github.com/neulab/Text-Summarization-Papers)
  - [Modern History for Text Summarization](http://pfliu.com/Historiography/summarization/summ-eng.html)



### Recommended Papers list

#### Review

| Year | Paper                                                        |
| ---- | ------------------------------------------------------------ |
| 2018 | [A Survey on Neural Network-Based Summarization Methods](https://arxiv.org/abs/1804.04589)<br />Y. Dong |
| 2020 | [Review of Automatic Text Summarization Techniques & Methods](https://pdf.sciencedirectassets.com/280416/AIP/1-s2.0-S1319157820303712/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAMaCXVzLWVhc3QtMSJIMEYCIQCqwas9C5XBrxGWAixtSVG1JHu4Ir1gH4OFpMeFjVcnxQIhAJnmwsesWxU2kSicjrm72Lw1TzC0I1PTDcwulAxemPzhKr0DCIv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAxoMMDU5MDAzNTQ2ODY1IgxjvutakJbTaBAnOb8qkQMCi48gc%2BweQ6ZCJWzaYTB4ucZyar0sFaZcnzb1wkymdAJwh9m1e%2BwLkZw2xJXLvFlUGn27lEgdYDbka8f%2BohT9oOOkF9QyGIen0yWhqlt4BB0jR6q2PyxdCswlFvY6VBuoK0g9%2Fm6oquTm37MbVHkqnaz70F%2Fy9xn5XpgjPRqrijfCP7Qf8Yd83kfWA7AQ3oxpXwIz8THWSwzlENkVBf8DByWAOvBnBnBD9K1keKjH%2FLQrCSkOgGuNOgaMPm%2FOiCzhRba4bYJJhZChjMcmNqxXczL8ebiCoIydZ923gygB5xDJpqEtP0vt0PpzEa6%2BKi03JJeXQDx3c0qQFejh52UkkqPps9jwF7dGejjgiR8WqNGWrJijW74u%2Bcys2y%2Fv8hcyME4mqlAfiXRPy0qyf6U3NA5EsaFSDR0DXR3bW39F8sIIRCeWOITf6q7rjExzvMdtr%2FsDdtKgghwR9PM75SyvX8FzYeCptHuoR3rfhc3RIxP96MNDdRIbGsht%2BJFuGUYYzuCwXPfUg%2B9eVRuUNT2bSzCPrpj%2BBTrqAep6mCVgTebUDbYKr1eHAVyOOzbsfz4lrlQN4jl3SyAFE%2FYMYxP0AyDB0rIRG8GjzfGKFzqQQScQ77d8m1ECTlFG2IuRqhvuWqIkYt21%2B3OJLSbFJ1kxhR8GLgi1%2BLYU2PJJQoVkhVbzeiPpAYh4vrjx3BdD1Y9xcGRkp5VP01DdkoYlbXpM4OkfTk6las12N8uZIfbSSqnfoepQO%2FunMSudM7nGOVphQU4TsRYDPtVYug1vy8mtj54GhcawwlcsaDPhF2tZ7hdEPyY%2BGSjyXU0ZMTffxJIhPMZUFFEtxjbmzRpSg3%2FEkKyQXQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20201201T105450Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY72IXVKMF%2F20201201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5f657b7900fbfc13936b686de2c66279a3ff74fbf1c0345191c2f0f68223e464&hash=3a55b9be508107240e832695ad5bfc371f18cc0dc0dcef5b45b1067da37346d6&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1319157820303712&tid=spdf-d04b0e55-eee7-401f-836b-11e1fc061edf&sid=2ae121a980d1464ffe6b55c8786454c8f4aagxrqa&type=client)<br />Widyassari, A. P., Rustad, S., Shidik, G. F., Noersasongko, E., Syukur, A., & Affandy, A. |
| 2020 | [**What Have We Achieved on Text Summarization?**](https://arxiv.org/abs/2010.04529)<br />Dandan Huang, Leyang Cui, Sen Yang, Guangsheng Bao, Kun Wang, Jun Xie, Yue Zhang / EMNLP |
| 2020 | [A Survey of Knowledge-Enhanced Text Generation](https://arxiv.org/abs/2010.04389)<br />Wenhao Yu, Chenguang Zhu, Zaitang Li, Zhiting Hu, Qingyun Wang, Heng Ji, Meng Jiang |



#### Classic

| Year | Model Name | Paper                                                        | Keywords |
| ---- | ---------- | ------------------------------------------------------------ | -------- |
| 1958 |            | [Automatic creation of literature abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf)<br />P.H. Luhn | gen-ext  |
| 2000 |            | [Headline Generation Based on Statistical Translation](http://www.anthology.aclweb.org/P/P00/P00-1041.pdf)<br />M. Banko, V. O. Mittal, and M. J. Witbrock | gen-abs  |
| 2004 | LexRank    | [LexRank: graph-based lexical centrality as salience in text summarization](https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)<br />G. Erkan, and D. R. Radev, | gen-ext  |
| 2005 |            | [Sentence Extraction Based Single Document Summarization](http://oldwww.iiit.ac.in/cgi-bin/techreports/display_detail.cgi?id=IIIT/TR/2008/97)<br />J. Jagadeesh, P. Pingali, and V. Varma | gen-ext  |
| 2010 |            | [Title generation with quasi-synchronous grammar](https://www.aclweb.org/anthology/D/D10/D10-1050.pdf)<br />K. Woodsend, Y. Feng, and M. Lapata, | gen-ext  |
| 2011 |            | [Text summarization using Latent Semantic Analysis](https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis)<br />M. G. Ozsoy, F. N. Alpaslan, and I. Cicekli | gen-ext  |



#### Based on Neural Net

| Year | Model Name | Paper                                                        | Keywords                                                     |
| ---- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2014 |            | [On using very large target vocabulary for neural machine translation](http://www.aclweb.org/anthology/P15-1001)<br />S. Jean, K. Cho, R. Memisevic, and Yoshua Bengio | gen-abs                                                      |
| 2015 | NAMAS      | [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685) ([Code](https://github.com/facebookarchive/NAMAS))<br />A. M. Rush, S. Chopra, and J. Weston / EMNLP | gen-abs<br />arch-att                                        |
| 2015 |            | [Toward Abstractive Summarization Using Semantic Representations](https://arxiv.org/pdf/1805.10399.pdf)<br/>Fei Liu,Jeffrey Flanigan,Sam Thomson,Norman M. Sadeh,Noah A. Smith / NAA-CL | gen-abs, task-event, arch-graph                              |
| 2016 |            | [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/pdf/1603.07252.pdf)<br/>Jianpeng Cheng,Mirella Lapata / ACL | gen-2stage                                                   |
| 2016 |            | [Abstractive sentence summarization with attentive recurrent neural networks](http://www.aclweb.org/anthology/N16-1012)<br />S. Chopra, M. Auli, and A. M. Rush / NAA-CL | gen-abs, arch-rnn, arch-cnn, arch-att                        |
| 2016 |            | [Abstractive text summarization using sequence-to-sequence RNNs and beyond](https://arxiv.org/abs/1602.06023)<br />R. Nallapati, B. Zhou, C. dos Santos, C. Gulcehre, and B. Xiang / CoNLL | gen-abs, data-new                                            |
| 2017 |            | [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230)<br />R. Nallapati, F. Zhai and B. Zhou | gen-ext<br />arch-rnn                                        |
| 2017 |            | [Get to the point: Summarization with pointergenerator networks](https://arxiv.org/abs/1704.04368) ([Code](https://github.com/abisee/pointer-generator))<br />A. See, P. J. Liu, and C. D. Manning | gen-ext/abs                                                  |
| 2017 |            | [A deep reinforced model for abstractive summarization](https://arxiv.org/abs/1705.04304)<br />R. Paulus, C. Xiong, and R. Socher | gen-ext/abs                                                  |
| 2017 |            | [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](https://pdfs.semanticscholar.org/c624/c38e53f321a6df2d16bd707499ce744ca114.pdf)<br/>Jiwei Tan,Xiaojun Wan,Jianguo Xiao / ACL | gen-ext, gen-abs, arch-graph, arch-att                       |
| 2017 |            | [Deep Recurrent Generative Decoder for Abstractive Text Summarization](https://arxiv.org/pdf/1708.00625.pdf)<br/>Piji Li,Wai Lam,Lidong Bing,Zihao W. Wang / EMNLP | latent-vae                                                   |
| 2017 |            | [Generative Adversarial Network for Abstractive Text Summarization](https://arxiv.org/abs/1711.09357) |                                                              |
| 2018 |            | [Controlling Decoding for More Abstractive Summaries with Copy-Based Networks](https://arxiv.org/abs/1803.07038)<br />N. Weber, L. Shekhar, N. Balasubramanian, and K. Cho | gen-ext/abs                                                  |
| 2018 |            | [**Generating Wikipedia by Summarizing Long Sequences**](https://arxiv.org/abs/1801.10198)<br />P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer / ICLR | gen-ext/abs                                                  |
| 2018 |            | [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704)<br />T. Baumel, M. Eyal, and M. Elhadad | gen-ext/abs                                                  |
| 2018 |            | [Bottom-Up Abstractive Summarization](https://arxiv.org/pdf/1808.10792.pdf)<br/>Sebastian Gehrmann,Yuntian Deng,Alexander M. Rush / EMNLP | gen-abs, arch-cnn, arch-att, eval-metric-rouge               |
| 2018 |            | [Deep Communicating Agents for Abstractive Summarization](https://arxiv.org/pdf/1803.10357.pdf)<br/>Asli Çelikyilmaz,Antoine Bosselut,Xiaodong He,Yejin Choi / **NAA-CL | gen-abs, task-longtext, arch-graph                           |
| 2018 |            | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080)<br />Y. Chen, M. Bansal | gen-ext/abs<br />arch-graph                                  |
| 2018 |            | [Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf)<br/>Shashi Narayan,Shay B. Cohen,Mirella Lapata | gen-ext, gen-abs, task-singledDoc, arch-rnn, arch-cnn, nondif-reinforce, eval-metric-rouge |
| 2018 |            | [BanditSum: Extractive Summarization as a Contextual Bandit](https://arxiv.org/pdf/1809.09672.pdf)<br/>Yue Dong,Yikang Shen,Eric Crawford,Herke van Hoof,Jackie Chi Kit Cheung | gen-ext, gen-abs, arch-rnn, nondif-reinforce, eval-metric-rouge |
| 2018 |            | [Content Selection in Deep Learning Models of Summarization](https://arxiv.org/pdf/1810.12343.pdf)<br/>Chris Kedzie,Kathleen McKeown,Hal Daumé | gen-ext, task-knowledge                                      |
| 2018 |            | [Faithful to the Original: Fact Aware Neural Abstractive Summarization](https://arxiv.org/pdf/1711.04434.pdf) |                                                              |
| 2018 |            | [A reinforced topic-aware convolutional sequence-to-sequence model for abstractive text summarization](https://www.ijcai.org/proceedings/2018/0619.pdf) |                                                              |
| 2018 |            | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization]() |                                                              |
| 2018 |            | [Global Encoding for Abstractive Summarization](https://arxiv.org/pdf/1805.03989.pdf) |                                                              |
| 2018 |            | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063) |                                                              |
| 2018 |            | [Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://www.aclweb.org/anthology/P18-1061) |                                                              |
| 2018 |            | [Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](https://aclweb.org/anthology/P18-1015) |                                                              |
| 2019 |            | [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318)<br />Y. Liu | gen-ext<br />                                                |
| 2019 |            | [Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/abs/1902.09243)<br />H. Zhang, J. Xu and J. Wang | gen-abs                                                      |
| 2019 |            | [Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization](https://arxiv.org/pdf/1906.00072.pdf)<br/>Sangwoo Cho,Logan Lebanoff,Hassan Foroosh,Fei Liu / ACL | task-multiDoc                                                |
| 2019 |            | [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/pdf/1905.06566.pdf)<br/>Xingxing Zhang,Furu Wei,Ming Zhou / ACL | arch-transformer                                             |
| 2019 |            | [ Searching for Effective Neural Extractive Summarization: What Works and What's Next](https://arxiv.org/pdf/1907.03491.pdf) Ming Zhong,Pengfei Liu,Danqing Wang,Xipeng Qiu,Xuanjing Huang / ACL | gen-ext                                                      |
| 2019 |            | [BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle](https://arxiv.org/pdf/1909.07405.pdf)<br/>Peter West,Ari Holtzman,Jan Buys,Yejin Choi / EMNLP | gen-ext, sup-sup, sup-unsup, arch-transformer                |
| 2019 |            | [Scoring Sentence Singletons and Pairs for Abstractive Summarization](https://arxiv.org/pdf/1906.00077.pdf)<br/>Logan Lebanoff,Kaiqiang Song,Franck Dernoncourt,Doo Soon Kim,Seokhwan Kim,Walter Chang,Fei Liu | gen-abs, arch-cnn                                            |
| 2019 |            | **[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)**([Code](https://github.com/google-research/pegasus))<br />Jingqing Zhang, Yao Zhao, Mohammad Saleh, Peter J. Liu / ICML 2020<br /><br />- [Review] 김한길. [영상](https://www.youtube.com/watch?v=JhGmeQBbDdA), [발표자료](https://www2.slideshare.net/hangilkim75/pegasus-237175343) |                                                              |
| 2020 |            | [TLDR: Extreme Summarization of Scientific Documents](https://arxiv.org/abs/2004.15011) ([Code](https://github.com/allenai/scitldr), [Demo](https://scitldr.apps.allenai.org/))<br />Isabel Cachola, Kyle Lo, Arman Cohan, Daniel S. Weld | gen-ext/abs                                                  |

#### References

- [neulab/Text-Summarization-Papers](https://github.com/neulab/Text-Summarization-Papers)
  - [10 must-read papers for neural **extractive** summarization](http://pfliu.com/pl-summarization/summ_paper_gen-ext.html)
  - [10 must-read papers for neural **abstractive** summarization](http://pfliu.com/pl-summarization/summ_paper_gen-abs.html)
- [icoxfog417/awesome-text-summarization](https://github.com/icoxfog417/awesome-text-summarization)

- [KaiyuanGao/awesome-deeplearning-nlp-papers](https://github.com/KaiyuanGao/awesome-deeplearning-nlp-papers)

