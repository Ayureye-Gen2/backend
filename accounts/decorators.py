from django.shortcuts import redirect

def authenticated_user(redirect_if_not_authenticated="login_page"):
    """
    In our case, the frontend is responsible for redirecting to login page if the use isn't authenticated for certain actions
    """
    def decorator(view_func):
        def wrapper_func(request, *args, **kwargs):
            if request.user.is_authenticated:
                return view_func(request, *args, **kwargs)
            return redirect(redirect_if_not_authenticated)
        return wrapper_func
    return decorator
