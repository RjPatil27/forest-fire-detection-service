import { Component, OnInit } from '@angular/core';
import { DashboardService } from '../../services/dashboard.service';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-model-page',
  templateUrl: './model-page.component.html',
  styleUrls: ['./model-page.component.scss']
})
export class ModelPageComponent implements OnInit {

  result: any;
  isRecieved: boolean = false;
  fileType: string = '';

  constructor(public dashboardService: DashboardService, private sanitizer: DomSanitizer) { }

  ngOnInit(): void {
  }

  onFileSelected(event: any) {
    this.isRecieved = false;
    const target = <HTMLInputElement>event.target;
    const files = <FileList>target.files;
    if (files) {
      const file = <File>files[0];
      this.fileType = file.type.includes('image') ? 'IMAGE' : 'VIDEO';
      this.dashboardService.uploadFile(this.fileType, file).subscribe(
        response => {
          // console.log(response);
          if (this.fileType === 'IMAGE') {
            const blob = new Blob([response], { type: 'application/image' });
            const unsafeImg = URL.createObjectURL(blob);
            this.result = this.sanitizer.bypassSecurityTrustUrl(unsafeImg);
            // console.log(this.resultImage);
            // read file as data url
            this.isRecieved = true;
          } else {
            const blob = new Blob([response], { type: 'application/video' });
            const unsafeVideo = URL.createObjectURL(blob);
            this.result = this.sanitizer.bypassSecurityTrustUrl(unsafeVideo);
            this.isRecieved = true;
          } 

        }
      );

    }
  }


}
