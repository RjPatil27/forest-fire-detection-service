import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { DashboardConstants } from '../../constants/dashboard-constants';
import { DashboardService } from '../../services/dashboard.service';

import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-scenario-page',
  templateUrl: './scenario-page.component.html',
  styleUrls: ['./scenario-page.component.scss']
})
export class ScenarioPageComponent implements OnInit {

  firstBatch = DashboardConstants.images['first-batch'];
  firstResult: any[] = [];
  secondBatch = DashboardConstants.images['second-batch'];
  secondResult: any[] = [];
  thirdBatch = DashboardConstants.images['third-batch'];
  thirdResult: any[] = [];

  showSpinner = false;

  constructor(public dashboardService: DashboardService, private sanitizer: DomSanitizer, private http: HttpClient) {

  }

  ngOnInit(): void {
  }

  sendImages(batch: string) {
    console.log('Inside')
    if (batch === 'first-batch') {
      for (let i = 0; i < DashboardConstants.images['first-batch'].length; i++) {
        this.http.get(this.firstBatch[i].src, { responseType: 'blob' })
          .subscribe(res => {
            var image = new File([res], this.firstBatch[i].src, {
              type: "image/png"
            });
            let formData = new FormData();
            formData.append('image', image, image.name);
            this.dashboardService.uploadImageFile(formData).subscribe(
              response => {
                const blob = new Blob([response], { type: 'application/image' });
                const unsafeImg = URL.createObjectURL(blob);
                const result = this.sanitizer.bypassSecurityTrustUrl(unsafeImg);
                // console.log(this.resultImage);
                // read file as data url
                this.firstResult.push({
                  'src': result
                }
                )
              }
            );
          });
      }
    } else if (batch === 'second-batch') {
      for (let i = 0; i < DashboardConstants.images['second-batch'].length; i++) {
        this.http.get(this.secondBatch[i].src, { responseType: 'blob' })
          .subscribe(res => {
            var image = new File([res], this.secondBatch[i].src, {
              type: "image/png"
            });
            let formData = new FormData();
            formData.append('image', image, image.name);
            this.dashboardService.uploadImageFile(formData).subscribe(
              response => {
                const blob = new Blob([response], { type: 'application/image' });
                const unsafeImg = URL.createObjectURL(blob);
                const result = this.sanitizer.bypassSecurityTrustUrl(unsafeImg);
                // console.log(this.resultImage);
                // read file as data url
                this.secondResult.push({
                  'src': result
                }
                )
              }
            );
          });
      }

    } else if (batch === 'third-batch') {
      for (let i = 0; i < DashboardConstants.images['third-batch'].length; i++) {
        this.http.get(this.thirdBatch[i].src, { responseType: 'blob' })
          .subscribe(res => {
            var image = new File([res], this.thirdBatch[i].src, {
              type: "image/png"
            });
            let formData = new FormData();
            formData.append('image', image, image.name);
            this.dashboardService.uploadImageFile(formData).subscribe(
              response => {
                const blob = new Blob([response], { type: 'application/image' });
                const unsafeImg = URL.createObjectURL(blob);
                const result = this.sanitizer.bypassSecurityTrustUrl(unsafeImg);
                // console.log(this.resultImage);
                // read file as data url
                this.thirdResult.push({
                  'src': result
                }
                )
              }
            );
          });
      }
    }

  }

}
