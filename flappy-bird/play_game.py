import time
import threading
from pynput import keyboard
from environ import FlappyBirdEnv

# --- Shared State for Keyboard Input ---
class HumanInput:
    def __init__(self):
        self.action = 0
        self.should_reset = False
        self.quit_requested = False

# --- Keyboard Listener Callbacks ---
def on_press(key, human_input):
    if key == keyboard.Key.space:
        human_input.action = 1
        human_input.should_reset = True

def on_release(key, human_input):
    if key == keyboard.Key.space:
        human_input.action = 0
        human_input.should_reset = False
    if key == keyboard.Key.esc:
        human_input.quit_requested = True
        return False  # Stop the listener

def main():
    """Main function to run the environment with human input."""
    env = FlappyBirdEnv(render_mode="human")
    human_input = HumanInput()

    # --- Start Keyboard Listener in a Background Thread ---
    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, human_input),
        on_release=lambda key: on_release(key, human_input)
    )
    listener.daemon = True
    listener.start()

    # --- Main Game Loop ---
    game_state = 'GAME_OVER'
    # obs, info = env.reset()
    print("Game ready. Press SPACE to start and flap. Press ESC to quit.")
    frame_count = 0
    running = True
    while running:
        # Always render the screen to keep the window responsive.
        env.render()

        action_to_take = 0

        if game_state == 'PLAYING':
            frame_count += 1
            action_to_take = human_input.action
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            if reward >= 1:
                frame_count = 0

            if terminated or truncated:
                print(f"Game Over. Score: {info['score']}. Frame Count : {frame_count}. Press SPACE to restart.")
                game_state = 'GAME_OVER'
                frame_count = 0

        elif game_state == 'GAME_OVER':

            env.step(0)

            # Check if the user wants to restart.
            if human_input.should_reset:
                obs, info = env.reset()
                game_state = 'PLAYING'
                human_input.should_reset = False
        
        # Check for quit signals from either the keyboard (ESC) or the window (X).
        if human_input.quit_requested or env.quit_signal:
            running = False
        

    # --- Cleanup ---
    env.close()
    print("Game closed.")

if __name__ == "__main__":
    main()
