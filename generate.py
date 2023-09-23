from utils.generator import GPTGenerator


if __name__ == "__main__":
    args = {
        'temperature': 1,
    }

    generator = GPTGenerator(data_set="vicuna80ar",
                             generator='gpt-3.5-turbo',
                             args=args,
                             user_name='',
                             n_processes=20,
                             save_for_eval=True)
    generator.generate()
    
    # generator = GPTGenerator(data_set="vicuna80ar",
    #                          generator='gpt-4-api-chatanywhere',
    #                          api_key='',
    #                          args=args,
    #                          n_processes=1,
    #                          save_for_eval=True)
    # generator.generate()




