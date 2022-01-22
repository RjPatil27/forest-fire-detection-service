import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

import { environment } from 'src/environments/environment';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DashboardService {

  baseUrl: string = environment.baseUrl;
  imageToShow: any;

  constructor(private http: HttpClient) { }

  uploadFile(fileType: string, file: File) {
    const formData = new FormData();
    formData.append(fileType.toLowerCase(), file, file.name);
    return fileType === 'IMAGE' ? this.uploadImage(formData) : this.uploadVideo(formData);
  }

  uploadImageFile(formData: FormData) {
    return this.uploadImage(formData);
  }

  uploadImage(formData: FormData)  {
    return this.http.post(this.baseUrl + '/upload' + '/image', formData, {responseType: 'blob'})
  }

  uploadVideo(formData: FormData)  {
    return this.http.post(this.baseUrl + '/upload' + '/video', formData, {responseType: 'blob'})
  }
}
