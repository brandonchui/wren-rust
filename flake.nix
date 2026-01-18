{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustc
            cargo
            rust-analyzer
            clippy
            llvmPackages_20.llvm
            libffi
            libxml2
          ];

          LLVM_SYS_201_PREFIX = "${pkgs.llvmPackages_20.llvm.dev}";
        };
      });
}
