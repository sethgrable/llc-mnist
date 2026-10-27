import copy
import math
import torch


# llc gets imported into run_llc
# Many argument values and funcs give in run_llc 

def estimate_llc(model, loader, loss_fn, n, epsilon=1e-5, gamma=100.0,
                 num_chains=4, num_steps=1000, burn_in=200, device="cpu"):
    nbeta = n / math.log(n)

    # w_star: a frozen copy of the trained weights w*. List of tensors, one per parameter
    w_star = [p.detach().clone() for p in model.parameters()]

    # L_n(w): average loss over the full dataset at w (w_star is not used here)
    with torch.no_grad():
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item() * len(x) #loss_fn returns the average loss over 
        loss_star = total / n                             # the batch 

# running the model four times starting with trained weights. 
# Calculating and storing losses with drift updates

    llcs, traces = [], []                  #Each trace is the list of losses recorded along one walk.
    for chain in range(num_chains):
        sampler = copy.deepcopy(model)     #copy.deepcopy copies everything inside too, so nothing is 
        losses = []                        #shared. Each walk starts at w*
        batches = iter(loader)             #loader in run_llc

        for step in range(num_steps):
            try:
                x, y = next(batches)
            except StopIteration:          # ran out of batches → start over
                batches = iter(loader)
                x, y = next(batches)
            x, y = x.to(device), y.to(device)

            loss = loss_fn(sampler(x), y)
            sampler.zero_grad()
            loss.backward()

            # SGLD step: drift toward low loss, pull back toward w*, plus noise
            with torch.no_grad():
                for p, p0 in zip(sampler.parameters(), w_star):
                    drift = -nbeta * p.grad - gamma * (p - p0)
                    p += 0.5 * epsilon * drift + math.sqrt(epsilon) * torch.randn_like(p)

            losses.append(loss.item())

        kept = losses[burn_in:]            # discard steps before the walk settles
        llcs.append(nbeta * (sum(kept) / len(kept) - loss_star))
        traces.append(losses)


    return llcs, traces, loss_star