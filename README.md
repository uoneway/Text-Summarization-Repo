# Text Summarization Repo
NLP 중 텍스트 요약 관련 자료를 축적해나가는 공간입니다. 
텍스트 요약 분야 공부를 시작하시는 분들에게 좋은 길잡이가 됐으면 합니다. 

  * [Prerequisite](#prerequisite)
  * [Resources](#resources)
    + [Must-read Papers](#must-read-papers)
    + [SOTA Models List](#sota-models-list)
  * [Data & Pre-trained Models](#data---pre-trained-models)
    + [Korean](#korean)
      - [Data&Competitions](#data-competitions)
      - [Pre-trained Models](#pre-trained-models)
    + [English](#english)
      - [Data&Competitions](#data-competitions-1)
  * [Others](#others)
    + [Resources](#resources-1)
    + [Recommended Papers list](#recommended-papers-list)

## Prerequisite

텍스트 요약 분야를 공부하는데 있어 알아두면 좋은 사전 지식과 추천 자료 목록입니다.

- NLP 기본 개념 이해

  - Embedding
  - Transfer learning(Pre-training  + Fine-tunning)

- Transformer/BERT 구조 및 Pre-training objective 이해

  최근에 나오는 NLP분야 논문들의 상당수가 Transformer에 기반하여 만들어진 BERT, 그리고 이 BERT의 변형인 RoBERTa, T5 등 여러 Pre-trained model에 기반하고 있습니다. 따라서 코드 수준의 세부적 이해까지는 아니더라도 이들의 개략적 구조와  Pre-training objective에 대한 이해를 가지고 있다면 논문을 읽거나 구현하는데 있어 큰 도움이 됩니다. 

  - [추천자료] [구상준(PINGPONG 블로그). Transformer - Harder, Better, Faster, Stronger: Transformer](https://blog.pingpong.us/transformer-review/)
  - [추천자료] [이유경(KoreaUniv DSBA) . Transformer to T5 (XLNet, RoBERTa, MASS, BART, MT-DNN,T5)](https://www.youtube.com/watch?v=v7diENO2mEA)

- Summarization task 기본 개념

  - 분류: Extractive/Abstractive, Multi/Single document 등
  - Metric: Rouge, BLEU, Perplexity(PPL) 등
  - Summarization 논문에서 자주 쓰이는 기본 용어: Gold/Oracle summary 등 

## Resources

### Must-read Papers

| Year | Model Name             | Paper                                                        | Keywords                                        | Code                                                 |
| ---- | ---------------------- | ------------------------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------- |
| 2004 | TextRank               | [Textrank: Bringing order into texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)<br />R. Mihalcea, P. Tarau<br />- [참고] [lovit. TextRank 를 이용한 키워드 추출과 핵심 문장 추출 (구현과 실험)](https://lovit.github.io/nlp/2019/04/30/textrank/) | gen-ext                                         | [lovit](https://github.com/lovit/textrank)           |
| 2019 | BertSum<br />(PreSumm) | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)<br/>Yang Liu,Mirella Lapata / EMNLP<br />- [참고] [이정훈(KoreaUniv DSBA) Paper Review](https://www.youtube.com/watch?v=PQk9kr9dGu0) | gen-ext/abs, <br />gen-2stage, arch-transformer | [Official](https://github.com/nlpyang/PreSumm)       |
| 2020 | MatchSum               | [Extractive Summarization as Text Matching](https://arxiv.org/abs/2004.08795)<br />Ming Zhong, Pengfei Liu, Yiran Chen, Danqing Wang, Xipeng Qiu, Xuanjing Huang / ACL<br />- [참고] [이유경(KoreaUniv DSBA) Paper Review](https://www.youtube.com/watch?v=8E2Ia4Viu94&t=1582s) | gen-ext                                         | [Official](https://github.com/maszhongming/MatchSum) |

### SOTA Models List

https://paperswithcode.com/task/text-summarization

https://www.paperdigest.org/2020/08/recent-papers-on-text-summarization/



## Data & Pre-trained Models

아래 사용한 약자의 의미는 다음과 같습니다.

* `w`: The number of words; `s`: The number of sentences
  예) `1000w -> 100w; 2s` 는 평균 1000개 단어의 문서와 이에 대한 평균 100개 단어이자 2개 문장 요약문이 제공된다는 의미입니다.
* `abs`: Abstractive summary; `ext`: Extractive summary

### Korean

#### Data&Competitions

| Korean Dataset                                               | Domain              | Volume    | License |
| ------------------------------------------------------------ | ------------------- | --------- | ------- |
| [모두의 말뭉치-문서 요약 말뭉치](https://corpus.korean.go.kr/) | 뉴스<br />          | 13,167    |         |
| [sae4K](https://github.com/warnikchow/sae4k)                 |                     | 50,000    |         |
| [sci-news-sum-kr-50](https://github.com/theeluwin/sci-news-sum-kr-50) | 뉴스(IT/과학)<br /> | 50        | MIT     |
| [Bflysoft-뉴스기사 데이터셋](https://dacon.io/competitions/official/235671/data/)<br />- [한국어 문서 추출요약 AI 경진대회(~ 2020.12.09)](https://dacon.io/competitions/official/235671/overview/)<br />- [한국어 문서 생성요약 AI 경진대회(~ 2020.12.09)](https://dacon.io/competitions/official/235673/overview/) | 뉴스<br />          | 43,000(p) |         |

#### Pre-trained Models

| Model                                                        | Pre-training Info.                                           | 이용                                                         | License                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- |
| [BERT(multilingual)](https://github.com/google-research/bert/blob/master/multilingual.md)<br /><br />BERT-Base(110M parameters) | - Wikipedia(multilingual)<br />- WordPiece. <br />- 110k shared vocabs | - [`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) 버전 권장<br />(`--do_lower_case=false` 옵션 넣어주기)<br />- Tensorflow | - Google<br />-Apache 2.0                      |
| [KOBERT](https://github.com/SKTBrain/KoBERT)<br />BERT-Base(92M parameters) | - 위키백과(문장 5M개), 뉴스(문장 20M개)<br />- [SentencePiece](https://github.com/google/sentencepiece)<br />- 8,002 vocabs(unused token 없음) | - PyTorch<br />- [KoBERT-Transformers(monologg)](https://github.com/monologg/KoBERT-Transformers)를 통해 <br />Huggingface Transformers 라이브러리 형태로 사용 및 [DistilKoBERT](https://github.com/monologg/DistilKoBERT) 이용 가능 | -SKTBrain<br />- Apache-2.0                    |
| [KorBERT](https://aiopen.etri.re.kr/service_dataset.php)<br />BERT-Base | - 뉴스(10년 치), 위키백과 등 23GB<br />- [ETRI 형태소분석 API](https://aiopen.etri.re.kr/service_api.php) / WordPiece(두 버전을 별도로 제공)<br />- 30,349 vocabs<br />- Latin alphabets: Cased<br />- [참고] [관련 발표자료](https://www2.slideshare.net/LGCNSairesearch/nlu-tech-talk-with-korbert) | - PyTorch, Tensorflow <br/>                                  | - ETRI<br />- 개별 협약 후 사용                |
| [KcBERT](https://github.com/Beomi/KcBERT)<br />BERT-Base/Large | - 네이버 뉴스 댓글(12.5GB, 8.9천만개 문장)<br />(19.01.01 ~ 20.06.15 기사 중 댓글 많은 기사 내 댓글과 대댓글)<br />- [tokenizers](https://github.com/huggingface/tokenizers)의 BertWordPieceTokenizer<br />- 30,000 vocabs |                                                              | - [Beomi](https://github.com/Beomi)<br />- MIT |
| [KoBART](https://github.com/SKT-AI/KoBART)<br />[**BART**](https://arxiv.org/pdf/1910.13461.pdf)(124M) | - 위키백과(5M), 기타(뉴스, 책, 모두의 말뭉치 (대화, 뉴스, ...), 청와대 국민청원 등 0.27B)<br />- [tokenizers](https://github.com/huggingface/tokenizers)의 Character BPE tokenizer<br />- 30,000 vocabs(<unused> 포함) | - 요약 task에 특화<br />- Huggingface Transformers 라이브러리 지원<br />- PyTorch | - SKT *T3K*<br />- modified MIT                |

- 기타
  - https://github.com/snunlp/KR-BERT



### English

#### Data&Competitions

[기타 요약 관련 영어 데이터셋 명칭, domain, task, paper 등](http://pfliu.com/pl-summarization/summ_data.html)

| English Dataset                                              | Domain / Length                                              | Volume                 | License                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| [ScisummNet](https://cs.stanford.edu/~myasu/projects/scisumm_net/)([paper](https://arxiv.org/abs/1909.01716))<br />*ACL(computational linguistics, NLP) research papers에 대해 세 가지 유형의 summary(논문 abstract, collection of citation sentences, human summary) 제공* <br />- CL-SciSumm 2019-Task2([repo](https://github.com/WING-NUS/scisumm-corpus), [paper](https://arxiv.org/abs/1907.09854))<br />- [CL-SciSumm @ EMNLP 2020-Task2](https://ornlcda.github.io/SDProc/sharedtasks.html#clscisumm)([repo](https://github.com/WING-NUS/scisumm-corpus)) | - Research paper<br />(computational linguistics, NLP)<br />- 4,417w → 110w; 2s; 151w | 1,000(abs/ ext)        | [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) |
| [LongSumm](https://github.com/guyfe/LongSumm)<br />*NLP 및 ML 분야 Research paper에 대해 상대적으로 장문의 summary(관련 blog posts 기반 abs, 관련 conferences videos talks 기반 ext)를 제공*<br />- [LongSumm 2020@EMNLP 2020](https://ornlcda.github.io/SDProc/sharedtasks.html#longsumm)<br />- [LongSumm 2021@ NAACL 2021](https://sdproc.org/2021/sharedtasks.html#longsumm) | - Research paper(NLP, ML)<br />- Long → abs(100-1,500 words); ext(30 sents / 990 words) | 700(abs) +  1,705(ext) | [Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| [CL-LaySumm](https://github.com/WING-NUS/scisumm-corpus/blob/master/README_Laysumm.md)<br />*NLP 및 ML 분야 Research paper에 대해 비전문가를 위한 쉬운(lay) summary 제공*<br />- [CL-LaySumm @ EMNLP 2020](https://ornlcda.github.io/SDProc/sharedtasks.html#laysumm) | - Research paper(epilepsy, archeology, materials engineering)<br />- Long → 70~100w | 600(abs)               | [a.dewaard@elsevier.com](mailto:a.dewaard@elsevier.com) 로 이메일을 보낸 후 contract 필요 |



## Others

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

#### Classic

| Year | Model Name | Paper                                                        | Keywords | Code |
| ---- | ---------- | ------------------------------------------------------------ | -------- | ---- |
| 1958 |            | [Automatic creation of literature abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf)<br />P.H. Luhn | gen-ext  |      |
| 2000 |            | [Headline Generation Based on Statistical Translation](http://www.anthology.aclweb.org/P/P00/P00-1041.pdf)<br />M. Banko, V. O. Mittal, and M. J. Witbrock | gen-abs  |      |
| 2004 | LexRank    | [LexRank: graph-based lexical centrality as salience in text summarization](https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)<br />G. Erkan, and D. R. Radev, | gen-ext  |      |
| 2005 |            | [Sentence Extraction Based Single Document Summarization](http://oldwww.iiit.ac.in/cgi-bin/techreports/display_detail.cgi?id=IIIT/TR/2008/97)<br />J. Jagadeesh, P. Pingali, and V. Varma | gen-ext  |      |
| 2010 |            | [Title generation with quasi-synchronous grammar](https://www.aclweb.org/anthology/D/D10/D10-1050.pdf)<br />K. Woodsend, Y. Feng, and M. Lapata, | gen-ext  |      |
| 2011 |            | [Text summarization using Latent Semantic Analysis](https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis)<br />M. G. Ozsoy, F. N. Alpaslan, and I. Cicekli | gen-ext  |      |



#### Based on Neural Net

| Year | Model Name | Paper                                                        | Keywords                                                     | Code                                                  |
| ---- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| 2014 |            | [On using very large target vocabulary for neural machine translation](http://www.aclweb.org/anthology/P15-1001)<br />S. Jean, K. Cho, R. Memisevic, and Yoshua Bengio | gen-abs                                                      |                                                       |
| 2015 | NAMAS      | [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)<br />A. M. Rush, S. Chopra, and J. Weston / EMNLP | gen-abs<br />arch-att                                        | [Official](https://github.com/facebookarchive/NAMAS)  |
| 2015 |            | [Toward Abstractive Summarization Using Semantic Representations](https://arxiv.org/pdf/1805.10399.pdf)<br/>Fei Liu,Jeffrey Flanigan,Sam Thomson,Norman M. Sadeh,Noah A. Smith / NAA-CL | gen-abs, task-event, arch-graph                              |                                                       |
| 2016 |            | [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/pdf/1603.07252.pdf)<br/>Jianpeng Cheng,Mirella Lapata / ACL | gen-2stage                                                   |                                                       |
| 2016 |            | [Abstractive sentence summarization with attentive recurrent neural networks](http://www.aclweb.org/anthology/N16-1012)<br />S. Chopra, M. Auli, and A. M. Rush / NAA-CL | gen-abs, arch-rnn, arch-cnn, arch-att                        |                                                       |
| 2016 |            | [Abstractive text summarization using sequence-to-sequence RNNs and beyond](https://arxiv.org/abs/1602.06023)<br />R. Nallapati, B. Zhou, C. dos Santos, C. Gulcehre, and B. Xiang / CoNLL | gen-abs, data-new                                            |                                                       |
| 2017 |            | [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230)<br />R. Nallapati, F. Zhai and B. Zhou | gen-ext<br />arch-rnn                                        |                                                       |
| 2017 |            | [Get to the point: Summarization with pointergenerator networks](https://arxiv.org/abs/1704.04368)<br />A. See, P. J. Liu, and C. D. Manning | gen-ext/abs                                                  | [GitHub](https://github.com/abisee/pointer-generator) |
| 2017 |            | [A deep reinforced model for abstractive summarization](https://arxiv.org/abs/1705.04304)<br />R. Paulus, C. Xiong, and R. Socher | gen-ext/abs                                                  |                                                       |
| 2017 |            | [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](https://pdfs.semanticscholar.org/c624/c38e53f321a6df2d16bd707499ce744ca114.pdf)<br/>Jiwei Tan,Xiaojun Wan,Jianguo Xiao / ACL | gen-ext, gen-abs, arch-graph, arch-att                       |                                                       |
| 2017 |            | [Deep Recurrent Generative Decoder for Abstractive Text Summarization](https://arxiv.org/pdf/1708.00625.pdf)<br/>Piji Li,Wai Lam,Lidong Bing,Zihao W. Wang / EMNLP | latent-vae                                                   |                                                       |
| 2017 |            | [Generative Adversarial Network for Abstractive Text Summarization](https://arxiv.org/abs/1711.09357) |                                                              |                                                       |
| 2018 |            | [Controlling Decoding for More Abstractive Summaries with Copy-Based Networks](https://arxiv.org/abs/1803.07038)<br />N. Weber, L. Shekhar, N. Balasubramanian, and K. Cho | gen-ext/abs                                                  |                                                       |
| 2018 |            | [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)<br />P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer | gen-ext/abs                                                  |                                                       |
| 2018 |            | [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704)<br />T. Baumel, M. Eyal, and M. Elhadad | gen-ext/abs                                                  |                                                       |
| 2018 |            | [Bottom-Up Abstractive Summarization](https://arxiv.org/pdf/1808.10792.pdf)<br/>Sebastian Gehrmann,Yuntian Deng,Alexander M. Rush / EMNLP | gen-abs, arch-cnn, arch-att, eval-metric-rouge               |                                                       |
| 2018 |            | [Deep Communicating Agents for Abstractive Summarization](https://arxiv.org/pdf/1803.10357.pdf)<br/>Asli Çelikyilmaz,Antoine Bosselut,Xiaodong He,Yejin Choi / **NAA-CL | gen-abs, task-longtext, arch-graph                           |                                                       |
| 2018 |            | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080)<br />Y. Chen, M. Bansal | gen-ext/abs<br />arch-graph                                  |                                                       |
| 2018 |            | [Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf)<br/>Shashi Narayan,Shay B. Cohen,Mirella Lapata | gen-ext, gen-abs, task-singledDoc, arch-rnn, arch-cnn, nondif-reinforce, eval-metric-rouge |                                                       |
| 2018 |            | [BanditSum: Extractive Summarization as a Contextual Bandit](https://arxiv.org/pdf/1809.09672.pdf)<br/>Yue Dong,Yikang Shen,Eric Crawford,Herke van Hoof,Jackie Chi Kit Cheung | gen-ext, gen-abs, arch-rnn, nondif-reinforce, eval-metric-rouge |                                                       |
| 2018 |            | [Content Selection in Deep Learning Models of Summarization](https://arxiv.org/pdf/1810.12343.pdf)<br/>Chris Kedzie,Kathleen McKeown,Hal Daumé | gen-ext, task-knowledge                                      |                                                       |
| 2018 |            | [Faithful to the Original: Fact Aware Neural Abstractive Summarization](https://arxiv.org/pdf/1711.04434.pdf) |                                                              |                                                       |
| 2018 |            | [A reinforced topic-aware convolutional sequence-to-sequence model for abstractive text summarization](https://www.ijcai.org/proceedings/2018/0619.pdf) |                                                              |                                                       |
| 2018 |            | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization]() |                                                              |                                                       |
| 2018 |            | [Global Encoding for Abstractive Summarization](https://arxiv.org/pdf/1805.03989.pdf) |                                                              |                                                       |
| 2018 |            | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063) |                                                              |                                                       |
| 2018 |            | [Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://www.aclweb.org/anthology/P18-1061) |                                                              |                                                       |
| 2018 |            | [Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization](https://aclweb.org/anthology/P18-1015) |                                                              |                                                       |
| 2019 |            | [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318)<br />Y. Liu | gen-ext<br />                                                |                                                       |
| 2019 |            | [Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/abs/1902.09243)<br />H. Zhang, J. Xu and J. Wang | gen-abs                                                      |                                                       |
| 2019 |            | [Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization](https://arxiv.org/pdf/1906.00072.pdf)<br/>Sangwoo Cho,Logan Lebanoff,Hassan Foroosh,Fei Liu / ACL | task-multiDoc                                                |                                                       |
| 2019 |            | [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/pdf/1905.06566.pdf)<br/>Xingxing Zhang,Furu Wei,Ming Zhou / ACL | arch-transformer                                             |                                                       |
| 2019 |            | [ Searching for Effective Neural Extractive Summarization: What Works and What's Next](https://arxiv.org/pdf/1907.03491.pdf) Ming Zhong,Pengfei Liu,Danqing Wang,Xipeng Qiu,Xuanjing Huang / ACL | gen-ext                                                      |                                                       |
| 2019 |            | [BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle](https://arxiv.org/pdf/1909.07405.pdf)<br/>Peter West,Ari Holtzman,Jan Buys,Yejin Choi / EMNLP | gen-ext, sup-sup, sup-unsup, arch-transformer                |                                                       |
| 2019 |            | [Scoring Sentence Singletons and Pairs for Abstractive Summarization](https://arxiv.org/pdf/1906.00077.pdf)<br/>Logan Lebanoff,Kaiqiang Song,Franck Dernoncourt,Doo Soon Kim,Seokhwan Kim,Walter Chang,Fei Liu | gen-abs, arch-cnn                                            |                                                       |
| 2020 |            | [TLDR: Extreme Summarization of Scientific Documents](https://arxiv.org/abs/2004.15011)<br />Isabel Cachola, Kyle Lo, Arman Cohan, Daniel S. Weld | gen-ext/abs                                                  | [GitHub](https://github.com/allenai/scitldr)          |
|      |            |                                                              |                                                              |                                                       |

#### References

- [neulab/Text-Summarization-Papers](neulab/Text-Summarization-Papers)
  - [10 must-read papers for neural **extractive** summarization](http://pfliu.com/pl-summarization/summ_paper_gen-ext.html)
  - [10 must-read papers for neural **abstractive** summarization](http://pfliu.com/pl-summarization/summ_paper_gen-abs.html)
- https://github.com/icoxfog417/awesome-text-summarization

- [KaiyuanGao](https://github.com/KaiyuanGao)/[awesome-deeplearning-nlp-papers](https://github.com/KaiyuanGao/awesome-dee- plearning-nlp-papers)