# Bakalářská práce repo

Repozitář obsahuje zdrojové kódy bakalářské práce na téma "Automatická detekce anomálií v signálech nitrolebního a arteriálního tlaku".

## Struktura repozitáře

- `anotator/` - anotační software pro anotaci dat
- `detection/` - metody detekce

Každá podsložka obsahuje vlastní `README.md`. Projekt využívá [Nix Flakes](https://nixos.org/) pro správu závislostí, je proto potřeba mít nainstalovaný Nix a direnv, povolené Flakes a pak povolit jednotlivé podsložky pomocí příkazu `direnv allow`. Pokud Nix nepoužíváte, nainstalujte si nutné Python knihovny obsažené v souboru `flake.nix` pomocí nástroje pip.

