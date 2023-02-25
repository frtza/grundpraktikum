# Vorlage für Praktikumsprotokolle

Dieser Ordner enthält die Struktur für Praktikumsprotokolle mit
allem, was in diesem Kurs besprochen wurde.

In Basisordner liegt der LaTeX-Header, Literaturdatenbank (`lit.bib`) sowie die der LaTeX-Header für matplotlib
und die matplotlibrc.
Außerdem haben wir in `programme.bib` die korrekten Quellen für die verwendete Software angegeben.

In dem Unterordner `vXXX` liegt dann ein Template für einen einzelnen Versuch,
welches für die jeweiligen Versuche kopiert und „ausgefüllt“ werden kann.


# Extensions

* Jupyter
* LaTeX Workshop
* Office Viewer
* Code Spell Checker
* German - Code Spell Checker

# Altprotokolle

* [Altprotokolle der Fachschaft](http://fachschaft-physik.tu-dortmund.de/wordpress/studium/praktikum/altprotokolle/) (Passwort: V302)
* [awesome-ap](https://nicoweio.github.io/awesome-ap/?utm_medium=discord&utm_campaign=toolbox-workshop)

# make

im Basisordner können die PDFs mithilfe von `make vXXX` erstellt werde.

Mit `make all` werden alle in der Makefile (all: ...) aufgeführten makefiles rekursiv ausgeführt.


# Inkscape

Mit dem Bildbearbeitungsprogramm [Inkscape](https://inkscape.org/de/) können einfach Vektorgrafiken von bspl. Skizzen aus Anleitungen erstellt werden.

1. Inkscape öffnen > Neues Dokument > Reiter: Datei > öffnen...
2. Seite der Versuchsanleitung auswählen, auf der sich die gewünschte Grafik befindet.
3. Alles bis auf die Grafik entfernen.
4. Bild auf die Grafikgröße zuschneiden:
   Dokumenteneinstellungen (Umstallt + Strg +D) > Benutzerdefiniert > [Rand wählen und sperren z.B. 5mm] > Seitengröße auf Zeichnungs- /Auswahlgröße
5. Grafik als .pdf im Bildordner speichern.

# mathpix

Mit [mathpix ](https://mathpix.com/)kann latex-Code von Dokumenten, Fotos und sogar handgeschriebenen Gleichungen erstellt werden.

1. mathpix starten
2. Strg + Alt + M
