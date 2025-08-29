
book_summaries_dict = {
    "1984": (
        "Romanul lui George Orwell descrie o societate distopică aflată sub controlul total al statului. "
        "Oamenii sunt supravegheați constant de „Big Brother”, iar gândirea liberă este considerată crimă. "
        "Winston Smith, personajul principal, încearcă să reziste acestui regim opresiv. "
        "Este o poveste despre libertate, adevăr și manipulare ideologică."
    ),
    "The Hobbit": (
        "Bilbo Baggins, un hobbit confortabil și fără aventuri, este luat prin surprindere atunci când este invitat "
        "într-o misiune de a recupera comoara piticilor păzită de dragonul Smaug. Pe parcursul călătoriei, el descoperă "
        "curajul și resursele interioare pe care nu știa că le are. Povestea este plină de creaturi fantastice, "
        "prietenii neașteptate și momente tensionate."
    ),
    "To Kill a Mockingbird": (
        "Harper Lee explorează teme de justiție și rasism în sudul Statelor Unite. Prin ochii lui Scout Finch, vedem "
        "lupta tatălui ei, Atticus, pentru adevăr și echitate. Este o lecție despre empatie și moralitate."
    ),
    "Harry Potter and the Sorcerer's Stone": (
        "Harry descoperă că este vrăjitor și merge la Hogwarts, unde își face prieteni și se confruntă cu un secret periculos. "
        "Povestea introduce teme de magie, loialitate și curaj."
    ),
    "All Quiet on the Western Front": (
        "Urmărim experiențele unui soldat german în Primul Război Mondial. Romanul descrie ororile războiului și impactul psihologic asupra tinerilor."
    ),
    "The Little Prince": (
        "O poveste poetică despre un mic prinț care explorează lumi diferite și întâlnește personaje simbolice. "
        "Este o reflecție profundă asupra iubirii, copilăriei și a ceea ce contează cu adevărat."
    ),
    "The Book Thief": (
        "Liesel, o fată care fură cărți în Germania nazistă, oferă o perspectivă umană asupra suferinței. "
        "Naratorul este Moartea. Temele sunt moartea, rezistența, și speranța."
    ),
    "Animal Farm": (
        "O fabulă politică în care animalele revoluționare creează o societate egală care se transformă în dictatură. "
        "Romanul evidențiază corupția puterii și manipularea maselor."
    ),
    "Pride and Prejudice": (
        "Elizabeth Bennet și domnul Darcy se confruntă cu prejudecăți și diferențe sociale. "
        "Romanul explorează dragostea, mândria și normele sociale din secolul al XIX-lea."
    ),
    "The Alchemist": (
        "Santiago pornește într-o călătorie spirituală către o comoară din Egipt, dar învață că adevărata comoară este descoperirea de sine."
    )
}


def get_summary_by_title(title: str) -> str:
    """
    Retrieve the summary of a book by its exact title.

    Looks up the given title inside `book_summaries_dict` (a dictionary mapping
    titles to summaries). If the title is not found, returns a default message
    in Romanian.

    Args:
        title (str): The exact book title to look up.

    Returns:
        str: The summary text if found, otherwise
             "Nu am gasit un rezumat pentru aceasta carte."
    """
    return book_summaries_dict.get(title, "Nu am gasit un rezumat pentru aceasta carte.")
