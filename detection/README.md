# Detekce

(Interaktivní) Python skript pro zvolené metody detekce v práci a pomocné skripty.

## Struktura
 - `bakalarka.py`
	 - (Interaktivní) Python skript (spouštěný přes ipython a vim-slime), rozdělen do strukturovaných buněk (podle komentáře). Skript vizualizuje a demonstruje metody předzpracování a transformace signálu. Dále se také věnuje testování metod detekce jako celku na dostupných datových sadách.
- `butter.py`
	- Pomocný skript pro vizualizaci Butterworthova filtru
- `check_dataset.py` 
	- Pomocný skript pro kontrolu datasetu, zda všechny `.hdf5` soubory mají korespondující `.artf` anotaci
