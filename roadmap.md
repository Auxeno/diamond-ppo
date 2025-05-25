## PPO Implementation Roadmap

### Nice readme
- Add image for plot timings

### Multi-discrete
- Support multi-discrete action spaces

### Recurrent PPO
- Make a Recurrent PPO implementation that supports truncations (can it be done?)

### Multi-agent PPO
- Independent PPO
- Centralised PPO
- True CTDE MAPPO

### Continuous?
- Consider whether we want to eventually support continuous action spaces
- Currently lean towards no, it's not where PPO's strength is

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