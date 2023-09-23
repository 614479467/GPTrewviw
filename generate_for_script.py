from utils.generator import GPTGenerator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--generator', type=str)
    parser.add_argument('--user_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--organization', type=str)
    parser.add_argument('--is_api_keys', action='store_true')
    parser.add_argument('--n_processes', type=int, default=10)
    args = parser.parse_args()
    
    gargs = {
        'temperature': args.temperature,
    }

    # generator = GPTGenerator(data_set=args.data_set,
    #                                  generator=args.generator,
    #                                  args=gargs,
    #                                  user_name=args.user_name,
    #                                  api_key=args.api_key,
    #                                  organization=args.organization,
    #                                  n_processes=args.n_processes,
    #                                  is_api_keys=args.is_api_keys,
    #                                  save_for_eval=False)
    # generator.generate()

    while True:
        try:
            generator = GPTGenerator(data_set=args.data_set,
                                     generator=args.generator,
                                     args=gargs,
                                     user_name=args.user_name,
                                     api_key=args.api_key,
                                     organization=args.organization,
                                     is_api_keys=args.is_api_keys,
                                     n_processes=args.n_processes,
                                     save_for_eval=False)
            generator.generate()
            break
        except:
            continue
  
