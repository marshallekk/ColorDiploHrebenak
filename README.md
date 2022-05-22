# ColorDiploHrebenak
Konzolová aplikácia na kolorizovanie videa alebo obrázkového vstupu

Užívateľská príručka:
1.  Stiahnutie prostredia Anaconda
2.  Vytvorte si nový enironment a nainstalujte nasledovne kniznice pomocou prikazov:
3.  conda create --name nazov
4.  conda activate nazov
5.  conda install numpy
6.  conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
7.  conda install matplotlib
8.  conda install scikit-image
9.  conda install tk
10.  conda install opencv
    
11. Dostaňte sa v prostredí anaconda k priečinku v ktorom máte stiahnuté toto riešenie
12. program spustíte príkazom: python colorMain.py

Pri spustení programu je potrebné zadať parametre:

  -video2frames  Konverzia videa na snímky
  
  -frames2video  Konverzia snímiek na video
  
  -complet       Kompletný proces konverzie videa na snímky, kolorizovanie
                 snímkov a následné spojenie snímkov do videa
                 
  -celeb         Načítanie checkpointu modelu trénovaného na datasete CelebA
  
  -places        Načítanie checkpointu modelu trénovaného na datasete
                 Places365
                 
  -placeleb      Načítanie checkpointu modelu trénovaného na kombinovanom
                 datasete Places365 a CelebA
                 
Ak chcete len kolorizovať obrázky, vložte obrázky (budú transformované na veľkosť prvého obrázka v priečinku) do priečinku kolorizuj/col

Pri parametri -complet je potrebné uviesť aj parameter modelu, ktorý chcete načítať

Príklad: python colorMain.py -complet -placeleb

Následne sa vykoná kompletný proces načítania videa, rozloženia na snímky, vyfarbenie snímkov, zloženie snímkov a uloženie videa za použitia modelu siete trénovanej na datasete Placeleb
