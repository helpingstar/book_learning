import numpy as np
import torch
from torch.distributions import Categorical

policy_net_output = torch.tensor([1, 1.5, 2, 2.5])
pdparams = policy_net_output
pd = Categorical(logits=pdparams)

print(pd.probs)
print(pd.logits)

print(torch.exp(pdparams))
print(torch.exp(pdparams)/torch.exp(pdparams).sum())
print(torch.log(pd.probs))

print(pd.log_prob(torch.tensor([0])))
# # 행동을 추출
# sample_count = np.zeros(2)
# for i in range(20000):
#     sample_count[pd.sample()] += 1

# print("Print Probability by sampling")
# print(sample_count)
# print(sample_count / sample_count.sum())
# # => [ 6060. 23940.]
# # => [0.202 0.798]

# print("Print Probability by calculate")
# print(torch.log(pdparams))
# print(torch.log(pdparams / pdparams.sum()))

# print(pdparams - pdparams.logsumexp(dim=-1, keepdim=True))

# # 행동 로그 확률을 계산
# print(pd.log_prob(action))
# # => tensor(-0.2231)
