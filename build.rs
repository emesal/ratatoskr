use vergen_gitcl::{Build, Cargo, Emitter, Gitcl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let build = Build::builder().build_timestamp(true).build();
    let cargo = Cargo::builder().build();
    let gitcl = Gitcl::builder().branch(true).sha(true).dirty(true).build();

    Emitter::default()
        .add_instructions(&build)?
        .add_instructions(&cargo)?
        .add_instructions(&gitcl)?
        .emit()?;

    // Compile protobuf when server or client feature is enabled
    #[cfg(any(feature = "server", feature = "client"))]
    {
        let proto_file = "proto/ratatoskr.proto";
        if std::path::Path::new(proto_file).exists() {
            tonic_build::configure()
                .build_server(cfg!(feature = "server"))
                .build_client(cfg!(feature = "client"))
                .compile_protos(&[proto_file], &["proto"])?;
        }
    }

    Ok(())
}
