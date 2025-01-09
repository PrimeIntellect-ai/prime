from prime_cli import get_availability, get_pods, set_api_key

# Set your API key
set_api_key("YOUR_API_KEY")

pods = get_pods()
print(pods)

availability = get_availability()
print(availability)
