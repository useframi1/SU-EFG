import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { ToastrService } from 'ngx-toastr';
import { catchError } from 'rxjs';

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const router = inject(Router);
  const toastr = inject(ToastrService);

  return next(req).pipe(
    catchError((error) => {
      if (error) {
        if (error.status === 0) {
          toastr.error('Could not connect to server', 'Server error');
        } else {
          toastr.error(error.error.detail, error.status);
        }
      }
      throw error;
    })
  );
};
