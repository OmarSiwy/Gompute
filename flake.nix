{
  description = "Gompute: Zig GPU compute library (CUDA + HIP)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      zig-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        # Zig master, 0.18.0-dev.349+d924f83bf at the locked overlay revision:
        # the same build ESPice pins. `build.zig.zon`'s minimum_zig_version and
        # `.github/workflows/*` name the same version; move all three together.
        zig = zig-overlay.packages.${system}.master;
        # The sandbox has no GPU and no nvidia-smi, so `.gpu = .auto` would
        # detect nothing. Nothing here calls emitKernels; anything that does
        # must pin `.gpu = .{ .name = "sm_89" }` to stay hermetic.
        zigBuild =
          name: step:
          pkgs.stdenvNoCC.mkDerivation {
            inherit name;
            src = self;
            nativeBuildInputs = [ zig ];
            dontConfigure = true;
            dontInstall = true;
            buildPhase = ''
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-cache"
              mkdir -p "$out"
              # Upstream's zig detects the native libc through a /usr/bin/env baked
              # into the binary; the sandbox has none, so it guesses musl and the
              # libc-linked tests cannot find /lib/ld-musl. Static musl needs no
              # loader. The glibc-only libm oracle tests skip here; they run in
              # the dev shell.
              zig build ${step} --prefix "$out" ${pkgs.lib.optionalString pkgs.stdenv.isLinux "-Dtarget=native-linux-musl"}
            '';
          };
      in
      {
        # `nix build` -> generated API docs; `nix flake check` -> zig build test.
        packages.default = zigBuild "gompute-docs" "docs";
        checks.default = zigBuild "gompute-test" "test";

        devShells.default = pkgs.mkShell {
          # Build only needs Zig; CUDA/HIP libraries are dlopen'd at run time.
          buildInputs = [ zig ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            "/run/opengl-driver"
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.rocm-runtime
          ];

          # One global cache per Zig store path: two builds of the same Zig
          # version (ESPice and VerA pin their own) sharing ~/.cache/zig evict
          # each other's build runner on every `zig build`.
          shellHook = ''
            export ZIG_GLOBAL_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/zig-${builtins.baseNameOf zig}"
            echo "Gompute dev shell: zig $(zig version)"
          '';
        };
      }
    );
}
