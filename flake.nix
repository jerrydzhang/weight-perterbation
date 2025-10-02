{
  description = "Python Development Enviornment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default =
          (pkgs.buildFHSEnv {
            name = "python-dev";

            targetPkgs = p:
              with p; [
                stdenv.cc.cc
                glibc
                zlib

                python3
                uv
              ];

            profile = ''
              unset PYTHONPATH
              uv sync --inexact
              uv pip install -e . -e "jernerics @ ./../jernerics"
              . .venv/bin/activate
            '';

            runScript = "zsh";
          }).env;

        hpc = pkgs.mkShell {
          packages = [
            pkgs.python3
            pkgs.uv
            pkgs.ruff
          ];

          shellHook = ''
            unset PYTHONPATH
            uv sync --inexact
            uv pip install -e . -e "jernerics @ ./../jernerics"
            . .venv/bin/activate
          '';
        };
      }
    );
  };
}
