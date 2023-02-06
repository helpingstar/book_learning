"""
Algorithm 2.2 : 이산 정책의 생성

1. 정책망 net, 분포 클래서 Categorical, 상태가 주어짐
2. pdaparams = net(state)를 계산
3. 행동 확률 분포의 인스턴스를 생성
    -> pd = Categorical(logits=pdparams)
4. pd를 이용하여 행동을 추출, action = pd.sample()
5. pd와 행동을 이용하여 행동 로그확률을 계산
    -> log_prob = pd.log_prob(action)
"""

import torch
from torch.distributions import Categorical

# 2개의 행동을 가정(카트폴: 왼쪽으로 이동, 오른쪽으로 이동)
# 정책 네트워크로부터 행동의 로짓 확률을 획득
policy_net_output = torch.tensor([-1.6094, -0.2231])
# pdparams는 로짓으로 probs = [0.2, 0.8]과 동일함
pdparams = policy_net_output
pd = Categorical(logits=pdparams)

# 행동을 추출
action = pd.sample()
print(action)
# => tensor(1) 또는 '오른쪽으로 이동'

# 행동 로그 확률을 계산
print(pd.log_prob(action))
# => tensor(-0.2231)
