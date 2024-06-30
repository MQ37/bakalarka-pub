{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    mynixpkgs.url = "github:MQ37/mynixpkgs";
  };
  outputs = { self, nixpkgs, flake-utils, mynixpkgs }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        with pkgs;
        {
          devShells.default = mkShell {
            buildInputs = [
              (python3.withPackages(ps: with ps; [


                # lsp
                jedi python-lsp-server

                ipython

                # anotator
                numpy h5py pyqtgraph pyside6

                # research
                pandas matplotlib scikit-learn
                mynixpkgs.mdp-toolkit
              ]))
            ];
          };
        }
      );
}

