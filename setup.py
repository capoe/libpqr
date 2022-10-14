from setuptools import setup, Extension, find_packages


class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def make_cxx_extensions():
    srcs = [
        "./libpqr/cxx/bindings.cpp",
        "./libpqr/cxx/gridsearch.cpp",
        "./libpqr/cxx/edgesearch.cpp"
    ]
    incdirs = [
        "./libpqr/cxx",
        get_pybind_include()
        #get_pybind_include(user=True)
    ]
    cpp_extra_link_args = []
    cpp_extra_compile_args = ["-std=c++11", "-O3"]
    c_extra_compile_args = ["-std=c99", "-O3"]
    return [
        Extension(
            'libpqr._cxx',
            srcs,
            include_dirs=incdirs,
            language='c++',
            extra_compile_args=cpp_extra_compile_args + ["-fvisibility=hidden"],  # the -fvisibility flag is needed by pybind11
            extra_link_args=cpp_extra_link_args,
        )
    ]

if __name__ == "__main__":
    setup(
        setup_requires=['pybind11>=2.4'],
        install_requires=['pybind11>=2.4'],
        ext_modules=make_cxx_extensions()
    )
