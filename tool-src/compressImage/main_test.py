
from utils.ripgrep import JsonModifier


if __name__ == '__main__':
    
    image_path_text = "test123"
    res_path = r"D:\yeh\dev\mcmod_dev\dianli_industrial\resource_pack_x4x17\ui"
    
    def _check(json_obj, path, value, file_path):

        return value

    modifier = JsonModifier.Instance()
    modifier.batch_modify_json_files(image_path_text, _check, res_path, whole_word=True)
    pass



