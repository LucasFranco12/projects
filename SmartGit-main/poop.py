## random pooop python program

##asci art poop

import time
import os
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_poop():
    poop = [
        "    💩    ",
        "   💩💩   ",
        "  💩💩💩  ",
        " 💩💩💩💩 ",
        "💩💩💩💩💩",
        " 💩💩💩💩 ",
        "  💩💩💩  ",
        "   💩💩   ",
        "    💩    "
    ]
    
    # Colors for the poop
    colors = [Fore.YELLOW, Fore.LIGHTYELLOW_EX, Fore.YELLOW, Fore.LIGHTYELLOW_EX]
    
    # Print the poop with colors
    for i, line in enumerate(poop):
        color = colors[i % len(colors)]
        print(color + line + Style.RESET_ALL)
        time.sleep(0.1)

def main():
    clear_screen()
    print(Fore.GREEN + "Generating poop..." + Style.RESET_ALL)
    time.sleep(1)
    clear_screen()
    print_poop()
    print("\n" + Fore.GREEN + "Poop generated successfully! 💩" + Style.RESET_ALL)

if __name__ == "__main__":
    main()

