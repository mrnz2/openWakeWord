import argparse
import glob
import shutil

from onnx2tf import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx", required=True)
    parser.add_argument("--output_tflite", required=True)
    parser.add_argument("--work_dir", required=True)
    args = parser.parse_args()

    convert(input_onnx_file_path=args.input_onnx, output_folder_path=args.work_dir)
    candidates = glob.glob(f"{args.work_dir}/**/*.tflite", recursive=True)
    if not candidates:
        raise RuntimeError("onnx2tf did not produce any .tflite file")
    shutil.copyfile(candidates[0], args.output_tflite)
    print(f"Saved: {args.output_tflite}")


if __name__ == "__main__":
    main()
