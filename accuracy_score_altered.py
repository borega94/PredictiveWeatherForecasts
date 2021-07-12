

def accuracy_score(gtruth, pred, normalize):
    """Diese Funktion ersetzt die accuracy score function von sckit learn
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    Args:
        gtruth (pd.series): Werte richtig
        pred (pd.series): Werte vorhergesagt
        normalize (boolean): Prozent oder Absolut
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    Returns:
        bool/int: Entweder absolute Anzahl oder in Prozent
    """
    anzahl_korrekt = 0
    anzahl = 0
    manuel_index = 0

    skip = False
    print(gtruth)
    print(pred)

    for i, rows in gtruth.iteritems():
        if manuel_index == (gtruth.size-1):
            skip = True

        if (rows == -1 & pred.iloc[manuel_index] == -1) & (rows == -1 & pred.iloc[manuel_index-1] == -1) & (rows == -1 & pred.iloc[manuel_index+1] == -1):
            anzahl = anzahl + 1

        if rows == 1 & pred.iloc[manuel_index] == 1:
            anzahl_korrekt = anzahl_korrekt + 1
        elif rows == 1 & pred.iloc[(manuel_index-1)] == 1:

            anzahl_korrekt = anzahl_korrekt + 1
            manuel_index = manuel_index + 1
        elif skip:
            break
        elif rows == 1 & pred.iloc[(manuel_index+1)] == 1:
            anzahl_korrekt = anzahl_korrekt + 1
            manuel_index = manuel_index + 1
        else:
            manuel_index = manuel_index + 1

        # print(manuel_index)

    print(anzahl)

    if normalize:
        acc = anzahl_korrekt / (gtruth.size-anzahl)
    else:
        acc = anzahl_korrekt

    print(anzahl_korrekt)
    print(gtruth.size-anzahl)
<<<<<<< Updated upstream
    return acc
=======
    return acc
>>>>>>> Stashed changes
