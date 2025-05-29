## PPO Implementation Roadmap

### Recurrent PPO Updates
- Make it easy to use LSTM or GRU

### Nice readme
- Add image for plot timings

### Multi-discrete
- Support multi-discrete action spaces

### Multi-agent PPO
- Independent PPO
- Centralised PPO
- True CTDE MAPPO

---

### Improve logger
- Verify custom logs logging works
- Logger should also be able to log other info like learning statistics (kl div, losses)
- Should be able to make nice graphs from training logs

### Check pixel obs work
- Actually check this (should be fine)

### Test performance with scripted torch
- Replace network with one that has jit compiled methods that go brr
- Check overall speedups

### Multi-threaded sync vector envs wrapper
- If you're feeling incredibly bored, make a multi-threaded vector envs wrapper. Should be quicker than the single-threaded and multi-processed version

### Magic numbers
- There are a few magic numbers still floating around, notably in utils.py classes. Put these in config

### More logging detail
- Consider do we want more advanced training stat loggin