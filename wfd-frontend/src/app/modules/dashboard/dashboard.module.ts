import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DashboardRoutingModule } from './dashboard-routing.module';

import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { ModelPageComponent } from './components/model-page/model-page.component';
import { ScenarioPageComponent } from './components/scenario-page/scenario-page.component';

import { MatTabsModule } from '@angular/material/tabs';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatCardModule } from '@angular/material/card';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatButtonModule } from '@angular/material/button';
import { MatDividerModule } from '@angular/material/divider';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { PredictionComponent } from './components/prediction/prediction.component';


@NgModule({
  declarations: [
    DashboardComponent,
    ModelPageComponent,
    ScenarioPageComponent,
    PredictionComponent
  ],
  imports: [
    CommonModule,
    DashboardRoutingModule,
    MatTabsModule,
    MatExpansionModule,
    MatCardModule,
    MatGridListModule,
    MatButtonModule,
    MatDividerModule,
    MatProgressSpinnerModule
  ],
})
export class DashboardModule { }
