import rogue_iface_tmux
import gpt
import time
import os

STEPS = 200
MEM_SIZE = 5

def term2str(term):
    return '\n'.join([''.join(row) for row in term])

def main():
    with open('kb.txt', 'r') as f:
        knowledge_base = f.read()

    game = rogue_iface_tmux.RogueInterface()
    game.restart()
    state = game.state()

    last_screens = []

    for i in range(STEPS):
        print(f'Step {i}')
        if not os.path.exists('states'):
            os.makedirs('states')

        with open(f'states/state_{i}.txt', 'w') as f:
            f.write(term2str(state))

        state = game.state()
        if len(last_screens) < MEM_SIZE:
            last_screens.append(state)
        else:
            last_screens.pop(0)
            last_screens.append(state)

        # descision = gpt.ask_o4_mini(
        #     prompt='You are playing the 1980s rogue game. ' +
        #            'Here are the game rules: \n' + knowledge_base + '\n' +
        #            'At the end of the prompt there will be a description of the current and few past game states. ' +
        #            'You need to choose which key to press to continue the game. ' +
        #            'You can press any key that exists in the game. ' +
        #            'Here are descriptions of the last game states, starting with old ones, the last frame is current:' +
        #            '\n#######################################################\n' +
        #            '\n#######################################################\n'.join([term2str(screen) for screen in last_screens]) +
        #            '\n#######################################################\n' +
        #            'I need ONLY the description of the key to press. For example, if you want to ' +
        #            'press "h" output exactly "h" and nothing else.',
        #     system_prompt='You are playing the 1980 rogue game. It is important for you to get through dungeons as far as possible.'
        # )

        descision = gpt.ask_o4_mini(
            prompt='Here are descriptions of the last dungeon states, starting with old ones, the last frame is current:' +
                   '\n#######################################################\n' +
                   '\n#######################################################\n'.join([term2str(screen) for screen in last_screens]) +
                   '\n#######################################################\n' +
                   'I need ONLY the description of the key to press. For example, if you want to ' +
                   'press "h" output exactly "h" and nothing else.',
            system_prompt=knowledge_base
        )

        if descision[:3].lower() == 'esc':
            key = chr(27)
        elif descision[:5].lower() == 'space':
            key = ' '
        else:
            key = descision

        if key[0].lower() == 'o':
            continue

        print(f'Key: {key}')

        game.key(key)

        if '--More--' in state[0]:
            game.key(' ')

        if 'Terse output' in state[0]:
            game.key(chr(27))
            game.key(' ')

        time.sleep(1)  # wait for the game to process the key

        state = game.state()
        print(term2str(state))

if __name__ == '__main__':
    main()