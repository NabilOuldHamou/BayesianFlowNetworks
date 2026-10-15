# BayesianFlowNetwork

Go read the paper it's peak https://arxiv.org/abs/2308.07037

This is here for the sole purpose of me avoiding having to look at the paper everytime I want to implement something from it.

The training loop is ordinary :
```py
model = DiscreteBFN(Network(), K=27, beta1=0.75)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.98))
for x in loader:
    loss = model.loss(x);
    opt.zero_grad();
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ema.update()   # paper uses EMA
```