## PPO Implementation Roadmap

### Document
- Config in particular

### Nice readme
- Swapping environment to pixel-based
- Shared policy and value functions
- Timer usage

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

### Check `np.as_array`
- When unpacking experience list, does np.as_array outperform np.array?

### Test performance with scripted torch
- Replace network with one that has jit compiled methods that go brr
- Check overall speedups

### Multi-threaded sync vector envs wrapper
- If you're feeling incredibly bored, make a multi-threaded vector envs wrapper. Should be quicker than the single-threaded and multi-processed version
