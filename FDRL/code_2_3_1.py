from torch.distributions import Normal
import torch

# 하나의 행동을 가정
# 정책 네트워크로부터 행동의 평균과 표준편차를 획득
policy_net_output = torch.tensor([1.0, 0.2])
# pdparams는 (평균, 표준편차) 또는 (loc, scale)
pdparams = policy_net_output
pd = Normal(loc=pdparams[0], scale=pdparams[1])

# 행동을 추출
action = pd.sample()
# => tensor(1.0295), 토크의 크기

# 행동 로그확률을 계산
print(pd.log_prob(torch.Tensor([1])))
# => tensor(0.6796)
