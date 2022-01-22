import { Component, OnInit } from '@angular/core';
import { environment } from 'src/environments/environment';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.scss']
})
export class PredictionComponent implements OnInit {

  ndvi: any = '';
  ndwi: any = '';

  loaded = false;

  baseUrl: string = environment.baseUrl;


  constructor(private http: HttpClient,  private sanitizer: DomSanitizer) { }

  ngOnInit(): void {

    this.http.get(this.baseUrl + '/get' + '/ndvi', {responseType: 'text'}).subscribe(
      response => {
        this.ndvi = this.sanitizer.bypassSecurityTrustHtml(response);
      }
    );
    this.http.get(this.baseUrl + '/get' + '/ndwi', {responseType: 'text'}).subscribe(
      response => {
        this.ndvi = this.sanitizer.bypassSecurityTrustHtml(response);
        this.loaded = true;
      }

    )
  }

}
