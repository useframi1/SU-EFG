import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { InstructionsComponent } from './instructions/instructions.component';
import { ManualEntriesComponent } from './manual-entries/manual-entries.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'instructions', component: InstructionsComponent },
  { path: 'manual-entries', component: ManualEntriesComponent },
  { path: '**', component: HomeComponent, pathMatch: 'full' },
];
