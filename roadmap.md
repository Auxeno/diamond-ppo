## PPO Implementation Roadmap

### Build logger
- Logger should record episode info, primarily
- Logger should also be able to log other info like learning statistics (kl div, losses)
- Should be able to print during training to console
- Should be able to make nice graphs from training logs

### Build checkpointer
- Checkpointer should basically just save model weights to disk periodically
- Logic to do this can be internal for max encapsulation

### Add timer
- Dig out timer from your old code, lets you separately profile different parts of the code
- Also prints logs nicely after a training run

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
