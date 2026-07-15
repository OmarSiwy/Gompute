{
  description = "Gompute — Zig GPU compute library (CUDA + HIP)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          # Build only needs Zig; CUDA/HIP libraries are dlopen'd at run time.
          buildInputs = with pkgs; [
            zig
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            "/run/opengl-driver"
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.rocm-runtime
          ];

          shellHook = ''
            echo "Gompute dev shell"
            echo "  zig: $(zig version)"
          '';
        };
      }
    );
}
