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

    Ok(())
}
