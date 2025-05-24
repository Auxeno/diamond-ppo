## PPO Implementation Roadmap

### Build checkpointer
- Checkpointer should basically just save model weights to disk periodically
- Logic to do this can be internal for max encapsulation

### Verify
- Verify GAE calculation is correct compared to CleanRL's. If it is that's awesome

### Document
- Add lots  more comments and documentation so code can be a much clearer reference for people learning

### Custom networks
- Make a nice way for users to use custom networks
- E.g. make it an optional argument when making a PPO object that overrides default logic

### Nice readme
- Basic usage guide
- Swapping out network
- Swapping environment to pixel-based
- Shared policy and value functions

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

### Check `np.as_array`
- When unpacking experience list, does np.as_array outperform np.array?

### Test performance with scripted torch
- Replace network with one that has jit compiled methods that go brr
- Check overall speedups
