fn main() {
    // Setzen wir die PyO3-Konfiguration
    println!("cargo:rustc-cfg=Py_3_7");
    println!("cargo:rustc-cfg=Py_3_8");
    println!("cargo:rustc-cfg=Py_3_9");
    println!("cargo:rustc-cfg=Py_3_10");
    println!("cargo:rustc-cfg=Py_3_11");
    println!("cargo:rustc-cfg=Py_3_12");
    println!("cargo:rustc-cfg=Py_LIMITED_API");
}
