# VWAP execution strategy with Soft Actor Critic, Proximal Policy Optimization
주식의 거래에 있어, 거래로 발생하는 시장충격에 의하여 거래비용이 발생하게 된다. 거래 규모가 클수록 이러한 거래비용은 증가하게 되며, 이를 줄이기 위하여 주문집행전략(Order Eexcution Strategy)을 사용하게 된다.

* TWAP (Time weighted average price)
* VWAP (Volume weighted average price)

## 거래량 프로파일링
VWAP 주문집행을 하는 전통적인 방식 중 하나는 다음과 같다.

* 대상 주식의 최근 장중 시간대별 거래량을 분석
* 분석자료를 바탕으로 시간대별 거래량 분포를 예측
* 예측자료에 근거하여 당일 시간대별 주문을 수행
딥러닝을 활용하여, 과거 주가자료를 분석하여 시간대별 거래량 분포를 예측하는 방법이 가능하다.

## 강화학습을 통한 VWAP 주문수행 구현
본 프로젝트는 VWAP을 구현하는데 있어 딥러닝 강화학습을 사용하여, 구현 가능성을 확인하고자 하는데 목적을 두었다.

LSTM with one time step, batch size = 1

* 학습된 모델을 실시간으로 활용하는 것을 고려하여 batch size = 1 로 테스트 하였다.
* 실제 HTS 상에서는 봉데이터가 완성되었을 때 결과를 바로 구할 수 있다.
* 액션은 정규분포를 통해 연속적인 값을 갖도록 하였다.

Advantage Actor-Critic, Generalized Advantage Estimator

* Temporal difference TD(0), TD(1) 테스트 후, 다양한 조건 테스트를 위해 GAE를 활용
* 본 테스트에서 gamma = 0.99, lambda = 0.95 를 사용하였다.

Proximal Policy Optimization & Entropy bonus for Exploration

* Local optimum 에 수렴하는 현상이 반복되어 Entropy bonus를 도입하였다.
* 보상함수는 주문잔량과 vwap 퍼포먼스를 고려하여 설계하였다. (Min -1 ~ Max 40)

입력데이터 - KODEX200 5분봉 데이터

* 종가, 종가-시가, 고가-저가, 거래량을 사용하였다.
* 현재가, 추세, 변동성, 거래량을 반영하고자 하였다.
* 시간정보를 반영하기 위해 [현재 스텝 / 전체 스텝]을 0 ~ 1 로 반영하였다.
* 주문잔량을 반영하기 위해, 초기값을 1로 한 후, 매 스텝 액션을 차감하여 주문잔량을 갱신하였다. 주문잔량은 1 ~ 0 으로 액션에 따라 점차 변하게 된다.

OpenAI Gym Env-Like State

* LSTM Network에 데이터 전달을 위해 OpenAI Gym Env와 유사한 구조를 만들어 활용하였다.
* env.step(action) 의 return의 구조는 다음과 같다.  
* state, reward, done, info
* state = [close, close-open, high-low, volume, current_bar, order_remained]
* info = (vwap performance ratio)


### 개선해야할 사항
* 학습이 원활하게 수렴하지 않음, Max reward 40 까지 도달하지 못함
* 주식의 가격, 거래량 관련 시계열 데이터의 예측 가능성이 높지 않음
* 그렇기 때문에 가격, 주문모델을 나눠서 구현하는 것이 바람직 할 것으로 생각됨
* Hyperparameter 튜닝 / learning rate, layer size, gamma, lambda, etc.
* LSTM의 time step을 변형하여 시도(5 step 정도)
* 보상함수 설계 개선


