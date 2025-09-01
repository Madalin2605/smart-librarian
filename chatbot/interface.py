from chatbot.agent import run_agent
import pyttsx3
from tools.language_filter import is_clean
from tools.image_generator import extract_chosen_title, generate_book_image


def speak_text(text):
    """
    Convert a given text string into spoken audio using pyttsx3.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    engine.say(text)
    engine.runAndWait()


def run_cli():
    """
    Simple Command-Line Interface (CLI) loop for Smart Librarian.
    Lets the user chat with the AI agent directly from the terminal.
    """
    print("📚 Bine ai venit la Smart Librarian!")
    print("💬 Pune o intrebare despre o carte sau scrie 'exit' pentru a iesi.")
    
    while True:
        user_input = input("\n Tu: ")
        if user_input.lower() in {"exit", "quit"}:
            print("La revedere!")
            break
        
        if not is_clean(user_input):
            print("Te rog pastreaza un limbaj respectuos. Iti pot recomanda carti pe orice tema.")
            continue

        print("🤖 Gandesc...")

        try:
            response = run_agent(user_input)
            print(f"\n Librarian:\n{response}")

            play = input("Vrei sa citeasca raspunsul? (y/n): ").strip().lower()
            if play == "y":
                speak_text(response)

            title = extract_chosen_title(response)
            if title:
                gen = input(f"Generez o ilustratie pentru „{title}”? (y/n): ").strip().lower()
                if gen == "y":
                    try:
                        image_path = generate_book_image(title, themes=None, size="1024x1024", lang="ro")
                        print(f"Imagine generata: {image_path}")
                    except Exception as e:
                        print(f"Eroare generare imagine: {e}")

        except Exception as e:
            print(f"Eroare: {e}")
