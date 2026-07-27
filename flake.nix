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
        # The sandbox has no GPU and no nvidia-smi, so `.gpu = .auto` would
        # detect nothing. Nothing here calls emitKernels; anything that does
        # must pin `.gpu = .{ .name = "sm_89" }` to stay hermetic.
        zigBuild =
          name: step:
          pkgs.stdenvNoCC.mkDerivation {
            inherit name;
            src = self;
            nativeBuildInputs = [ pkgs.zig ];
            dontConfigure = true;
            dontInstall = true;
            buildPhase = ''
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-cache"
              mkdir -p "$out"
              zig build ${step} --prefix "$out"
            '';
          };
      in
      {
        # `nix build` -> generated API docs; `nix flake check` -> zig build test.
        packages.default = zigBuild "gompute-docs" "docs";
        checks.default = zigBuild "gompute-test" "test";

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
