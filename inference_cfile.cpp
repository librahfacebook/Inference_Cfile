#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <typeinfo>
#include <memory>
#include <string.h>
#include <time.h>

using namespace std;

// global const
const int LEN_TEMPORAL = 32;       // the input image data size that processed
const int RESOLUTION_WIDTH = 384;  // the width of image
const int RESOLUTION_HEIGHT = 224; // the height of image

// create a list container for tensors
typedef std::list<torch::Tensor> TENSOR_LIST;
// create a map container for saliency map
typedef std::map<int, float **> SALIENCY_MAP;
// the jit script module
typedef torch::jit::script::Module MODULE;


/**
 * convert the output tensor into vector type
 * */
std::vector<std::vector<float>> convert_cpu_tensor_to_vector(torch::Tensor output) 
{
    std::vector<std::vector<float>> data_vector;
    auto output_float = output.accessor<float,3>();

    for(int i = 0; i < output_float.size(1); i++){
        std::vector<float> row_vector;
        for(int j = 0; j < output_float.size(2); j++){
            row_vector.push_back(output_float[0][i][j]);
        }
        data_vector.push_back(row_vector);
    }

    return data_vector;
}


/**
 * calculate the sum value of one region
 * */
float calculate_sum(std::vector<std::vector<float>> data_vector, int region_width_start, int region_width_end, int region_height_start, int region_height_end)
{
    float sum = 0.0;
    
    for (int i = region_height_start; i < region_height_end; i++)
    {
        for (int j = region_width_start; j < region_width_end; j++)
        {
            // convert tensor entry to float type
            float value = data_vector[i][j];
            sum += value;
        }
    }
    return sum;
}

/**
 * get the normalized map
 * */
float **get_normalized_map(float **saliency_map, int tile_width, int tile_height)
{
    float total = 0.0;
    for (int i = 0; i < tile_height; i++)
    {
        for (int j = 0; j < tile_width; j++)
        {
            total += saliency_map[i][j];
        }
    }

    for (int i = 0; i < tile_height; i++)
    {
        for (int j = 0; j < tile_width; j++)
        {
            saliency_map[i][j] = saliency_map[i][j] / total;
        }
    }

    return saliency_map;
}

/**
 * given the tile's width and height, and generate the saliency map.
 * */
float **get_saliency_map(torch::Tensor output, int tile_width, int tile_height)
{
    // get the rows and columns of the output
    int rows = torch::size(output, 1);
    int columns = torch::size(output, 2);

    // convert the tensor type to CPU tensors
    output = output.to(torch::kCPU);

    std::vector<std::vector<float>> data_vector = convert_cpu_tensor_to_vector(output);
    // std::cout << "size: " << data_vector.size() << std::endl;

    // initialize the saliency_map
    float **saliency_map = new float *[tile_height];
    for (int i = 0; i < tile_height; i++)
    {
        saliency_map[i] = new float[tile_width];
    }

    // process the raw output
    float sum = 0.0;

    int region_width_length = int(columns / tile_width);
    int last_region_width_length = columns - region_width_length * (tile_width - 1);

    int region_height_length = int(rows / tile_height);
    int last_region_height_length = rows - region_height_length * (tile_height - 1);

    for (int i = 0; i < tile_height; i++)
    {
        // get the region height index
        int region_height_start = i * region_height_length;
        int region_height_end;
        if (i == tile_height - 1)
            region_height_end = region_height_start + last_region_height_length;
        else
            region_height_end = region_height_start + region_height_length;

        for (int j = 0; j < tile_width; j++)
        {
            // get the sum value of region

            // get the region width index
            int region_width_start = j * region_width_length;
            int region_width_end;
            if (j == tile_width - 1)
                region_width_end = region_width_start + last_region_width_length;
            else
                region_width_end = region_width_start + region_width_length;

            sum = calculate_sum(data_vector, region_width_start, region_width_end, region_height_start, region_height_end);

            saliency_map[i][j] = sum;
    
        }
    }

    saliency_map = get_normalized_map(saliency_map, tile_width, tile_height);

    return saliency_map;
}

/**
 * get the inference result of salient network
 * inputs: input data tensor
 * module: the torch script module
 * */
torch::Tensor get_inference_result(std::vector<torch::jit::IValue> inputs, MODULE module)
{
    // define the return vlaue tensor type
    torch::Tensor output;

    // clock_t time_start = clock();
    // Execute the model and turn its output into a tensor.
    output = module.forward(inputs).toTensor();

    // clock_t time_end = clock();
    // std::cout << "inference time: " << time_end - time_start << "ms" << std::endl;

    return output;
}

/**
 * process the raw image data, and convert the images to tensor
 * */
torch::Tensor transform_image(cv::String img_path)
{
    // read image using opencv2
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(RESOLUTION_WIDTH, RESOLUTION_HEIGHT));

    // convert image to tensor
    torch::Tensor img_tensor = torch::from_blob(img_resize.data, {img_resize.rows, img_resize.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});

    // normalization
    img_tensor = img_tensor.toType(torch::kFloat32);

    img_tensor = img_tensor.div(255);
    img_tensor = (img_tensor - 0.5) / 0.5;
    img_tensor = img_tensor.unsqueeze(0);

    return img_tensor;
}

/**
 * inference the result using trained model and get saliency map
 * */
float **inference_saliecey_map(torch::Tensor input_tensor, MODULE module, int tile_width, int tile_height)
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(input_tensor);

   

    // Execute the model and turn its output into a tensor.

    torch::Tensor output = get_inference_result(inputs, module);

    //  clock_t time_start2 = clock();

    float **saliency_map = get_saliency_map(output, tile_width, tile_height);

    // clock_t time_end2 = clock();

    // std::cout << "inference saliency map time: " << time_end2 - time_start2 << "ms" << std::endl;

    return saliency_map;
}

/**
 * print the result for one saliency map
 * */
void print_map(float **saliency_map, int tile_width, int tile_height, int index)
{
    std::cout << "Frame " << index + 1 << " Saliency Map:\n";
    for (int i = 0; i < tile_height; i++)
    {
        for (int j = 0; j < tile_width; j++)
        {
            printf("%.6f ", saliency_map[i][j]);
            // std::cout << saliency_map[i][j] <<" ";
        }
        std::cout << endl;
    }
}

/**
 * print the result for all saliency maps
 * */
void print_maps(SALIENCY_MAP saliency_maps, int tile_width, int tile_height)
{
    // int maps_size = saliency_maps.size();
    
    // for (int i = 0; i < maps_size; i++)
    // {
    //     float **saliency_map = saliency_maps[i];
    //     print_map(saliency_map, tile_width, tile_height, i);
    // }

    // change the traverse method to iterator
    SALIENCY_MAP::iterator map_iter;
    for(map_iter=saliency_maps.begin(); map_iter!=saliency_maps.end();map_iter++){
        int index = map_iter->first;
        float **saliency_map = map_iter->second;
        print_map(saliency_map, tile_width, tile_height, index);
    }
}

/**
 * load the image datasets to inference
 * */
torch::Tensor listTotensor(TENSOR_LIST tensor_list)
{
    // build an list iterator
    TENSOR_LIST::iterator list_iterator;

    // build an images tensor(array type)
    torch::Tensor tensor_array[LEN_TEMPORAL];
    int i = 0;

    // traverse the list of tensors
    for (list_iterator = tensor_list.begin(); list_iterator != tensor_list.end(); list_iterator++)
    {
        tensor_array[i++] = (torch::Tensor)*list_iterator;
    }

    torch::Tensor input_data = torch::cat(tensor_array, 0);

    // std::cout << "data type:" << typeid(img_resize).name() << endl;

    // cv::imshow("img", img);
    // cv::waitKey();

    // process the input data tensor
    input_data = input_data.unsqueeze(0);
    input_data = input_data.permute({0, 2, 1, 3, 4});

    // std::cout << "shape:" << torch::size(input_data, 0) <<endl;

    return input_data;
}

/**
 * given the images root of one video, and produce the final saliency maps
 * */
SALIENCY_MAP produce_saliency_maps(string images_root, string weights_file, int tile_width, int tile_height, bool device_type)
{
    vector<cv::String> images_paths;

    float **saliency_map;

    // define the map for saving saliency maps of one video
    SALIENCY_MAP saliency_maps;

    // get the paths of all images using cv::glob
    cv::glob(images_root, images_paths);

    // define the tensor list for saving image tensor
    TENSOR_LIST tensor_list;

    // change the tensor_list to tensor type
    torch::Tensor input_data;

    // call the script module with weights_file
    MODULE module;
    try
    {
        module = torch::jit::load(weights_file);
        // assign module to gpu device, otherwise to cpu.
        if(device_type)
            module.to(torch::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the module\n";
        exit(1);
    }

    // process in a sliding window fashion
    for (int i = 0; i < images_paths.size(); i++)
    {
        // assign the input data on gpu device, otherwise on cpu device.
        torch::Tensor image_tensor = transform_image(images_paths[i]);
        if(device_type)
            image_tensor = image_tensor.to(torch::kCUDA);

        tensor_list.push_back(image_tensor);

        if (i >= LEN_TEMPORAL - 1)
        {
            // inference the last frame's saliency map according to the past LEN_TEMPORAL-1 frame
            input_data = listTotensor(tensor_list);
            saliency_map = inference_saliecey_map(input_data, module, tile_width, tile_height);
            saliency_maps[i] = saliency_map;

            // print_map(saliency_map, tile_width, tile_height, i);

            // process the first(len_temporal-1) frame
            if (i < 2 * LEN_TEMPORAL - 2)
            {
                // inference the first frame's saliency map according to the last LEN_TEMPORAL-1 frame
                input_data = listTotensor(tensor_list);
                // reverse the input data to inference the first frame's saliency map
                input_data = torch::flip(input_data, 2);
                saliency_map = inference_saliecey_map(input_data, module, tile_width, tile_height);
                saliency_maps[i - LEN_TEMPORAL + 1] = saliency_map;

                // print_map(saliency_map, tile_width, tile_height, i - LEN_TEMPORAL + 1);
            }
            tensor_list.pop_front();
        }
    }

    return saliency_maps;
}

/**
 * show the usage info
 * */
void showUsage()
{
    std::cout << "Usage    : rec_user_arg <--name=your name> [Option]" << endl;
    std::cout << "Options  :" << endl;
    std::cout << " --images_root=your images' root         The images' root is all images of one video, and it should be given otherwise default." << endl;
    std::cout << " --weights_file=your weights' file       The weights' file is the trained network parameters, and it should be given otherwise default." << endl;
    std::cout << " --tile_width=your tile's width          The saliency map's width, and it should be given otherwise default." << endl;
    std::cout << " --tile_height=your tile's height        The saliency map's height, and it should be given otherwise default." << endl;
    std::cout << " --verbose=(true or false)               show all saliency maps, and it should be given otherwise default false." << endl;

    return;
}

/**
 * commit the command and return results
 * */
void commit_command(int argc, char *argv[])
{
    string images_root, weights_file;
    int tile_width, tile_height;
    bool device_type, verbose;

    int nOptionIndex = 1;
    if (argc < 2)
    {
        std::cout << "No arguments, all values are given default!" << endl;
        showUsage();

        images_root = "../images";
        weights_file = "../convert_weights.pt";
        tile_width = 6;
        tile_height = 4;
        verbose = true;
    }
    else
    {
        while (nOptionIndex < argc)
        {
            if (strncmp(argv[nOptionIndex], "--images_root=", 14) == 0)
            {
                images_root = &argv[nOptionIndex][14];
            }
            else if (strncmp(argv[nOptionIndex], "--weights_file=", 15) == 0)
            {
                weights_file = &argv[nOptionIndex][15];
            }
            else if (strncmp(argv[nOptionIndex], "--tile_width=", 13) == 0)
            {
                tile_width = atoi(&argv[nOptionIndex][13]);
            }
            else if (strncmp(argv[nOptionIndex], "--tile_height=", 14) == 0)
            {
                tile_height = atoi(&argv[nOptionIndex][14]);
            }
            else if (strncmp(argv[nOptionIndex], "--verbose=", 10) == 0)
            {
                char *verbose_info = &argv[nOptionIndex][10];
                if (strcmp(verbose_info, "true") == 0)
                    verbose = true;
                else if (strcmp(verbose_info, "false") == 0)
                    verbose = false;
                else
                    std::cout << "The verbose info is not valid." << endl;
            }
            else if (strncmp(argv[nOptionIndex], "--help", 6) == 0)
            {
                showUsage();
                return;
            }
            else
            {
                std::cout << "Options '" << argv[nOptionIndex] << "' not valid. Run '" << argv[0] << "' for details." << endl;
                return;
            }
            nOptionIndex++;
        }
    }

    // auto choose the device type(cpu or gpu)
    if(torch::cuda::is_available()){
        device_type = true;
    }else{
        device_type = false;
    }

    clock_t time_start = clock();
    SALIENCY_MAP saliency_maps = produce_saliency_maps(images_root, weights_file, tile_width, tile_height, device_type);

    // show saliency maps
    if (verbose)
        print_maps(saliency_maps, tile_width, tile_height);

    clock_t time_end = clock();
    std::cout << "total process time: " << time_end - time_start << "ms" << std::endl;

    // return saliency_maps;
}

int main(int argc, char *argv[])
{
    commit_command(argc, argv);

    return 0;
}