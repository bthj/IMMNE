from gym.envs.registration import register

register(
    id="note_exploration/NoteWorld-v0",
    entry_point="note_exploration.envs:NoteWorldEnv",
)
