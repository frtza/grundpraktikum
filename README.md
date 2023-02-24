# Protokolle zum Grundpraktikum für Physikstudierende

**Beschreibung**

Aus [Moodle](https://moodle.tu-dortmund.de/):
> Zur Ausbildung eines Physik Studierenden an der [TU-Dortmund](https://www.tu-dortmund.de/) gehört im
> Bachelor Studiengang ein zweisemestriges Grundpraktikum, in dem an einfachen Standardversuchen unter
> anderem experimentelle Methoden in der Physik, Fehlerrechnung, das Schreiben von Protokollen sowie
> der Umgang mit physikalischen Geräten und Daten gelernt wird.

Die Struktur dieses Projekts und die grundlegende Methodik sind an den
[Toolbox-Workshop](https://toolbox.pep-dortmund.org/notes.html) von
[PeP et al. e.V.](https://pep-dortmund.org/) angelehnt. Als Hilfe stellt die
[Fachschaft](https://fachschaft-physik.tu-dortmund.de/wordpress/studium/praktikum/altprotokolle/)
Altprotokolle zur Verfügung.

**Autoren**

Fritz Agildere ([fritz.agildere@udo.edu](mailto:fritz.agildere@udo.edu)) und
Amelie Strathmann ([amelie.strathmann@udo.edu](mailto:amelie.strathmann@udo.edu))

**Struktur**

Die Protokolle werden mit `make` als PDF-Datei ausgegeben. Im Hauptverzeichnis wird die allgemeine Konfiguration
vorgenommen. Die Unterverzeichnisse übernehmen diese standardmäßig. Die einzelnen Versuche enthalten wiederum die
Verzeichnisse `build`, in dem sich alle generierten Dateien befinden, und `content`, das der Struktur des Protokolls
entspricht:

1. Zielsetzung
2. Theorie
3. Durchführung
4. Auswertung
5. Diskussion

Zur graphischen Darstellung und um abgeleitete Messwerte automatisch zu berechnen, werden `python` Skripte
mit den entsprechenden Bibliotheken genutzt. Die Dokumente werden unter Anwendung von `lualatex` kompiliert.

Das Projekt *Grundpraktikum* ist mit GNU/Linux kompatibel.
