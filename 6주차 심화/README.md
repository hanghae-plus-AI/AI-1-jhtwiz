질환에 대한 설명을 해주는 챗봇을 만든다고 가정하였습니다.

데이터셋은 AI Hub에서 받은 초거대 AI 헬스케어 질의응답 데이터 중 질환별 정의 데이터 중에서 일부 추출하여서 만들었습니다.(출처 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71762)

모델은 Gemma-2b로 진행하려 했으나 GPU 메모리 문제로 계속 실패하여 GPT2로 진행했습니다.

추출된 데이터셋은 25319개의 질문 답변으로 구성되어 있습니다.

300 step마다 train loss와 eval loss를 측정했습니다.

Train loss

Eval loss
