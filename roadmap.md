## PPO Implementation Roadmap

### Multi-discrete
- Support multi-discrete action spaces

---

### Check pixel obs work
- Actually check this (should be fine)

### Test performance with scripted torch
- Replace network with one that has jit compiled methods that go brr
- Check overall speedups

### Multi-threaded sync vector envs wrapper
- If you're feeling incredibly bored, make a multi-threaded vector envs wrapper. Should be quicker than the single-threaded and multi-processed version

### Magic numbers
- There are a few magic numbers still floating around, notably in utils.py classes. Put these in config
