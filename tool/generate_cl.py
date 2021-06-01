'''
This python script is to generate buildin kernel source file,
if you change or add the kernel source *.cl, please run this script and
generate new math21_opencl_kernel_source.h
'''
import os
import ntpath
import shutil
HEADER_FILE_NAME = "math21_opencl_kernel_source.h"

def extension(file_path):
    return os.path.splitext(file_path)[1]

def main():
    files = [os.path.join(root, file) for root, dirs, files in os.walk("..") for file in files if extension(file) == ".cl" or extension(file) == ".kl"]
    # print(files)

    with open(HEADER_FILE_NAME, "w") as des:
        des.writelines("#pragma once\n")
        des.writelines("#include<map>\n")
        for file in files:
            name = ntpath.basename(file)
            source_name = "std::string " + name[:-3] + " = \"\\n\\\n"
            des.writelines(source_name)
            with open(file, "r") as src:
                while True:
                    line = src.readline()
                    if line == "":
                        break
                    else:
                        line = line.replace("\n","") + "\\n\\\n"
                        des.writelines(line)
            end = "\"; \n"
            des.writelines(end)

        des.writelines("std::map<std::string,std::string> source_map = {\n")
        for file in files:
            name = ntpath.basename(file)
            des.writelines("std::make_pair(\"" + name + "\"," + name[:-3] + "),\n")
        des.writelines("};\n")
    shutil.move(HEADER_FILE_NAME,"../includes/"+HEADER_FILE_NAME)

if __name__ == "__main__":
    main()